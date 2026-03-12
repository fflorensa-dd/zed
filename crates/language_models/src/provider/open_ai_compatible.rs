use anyhow::{Result, anyhow};
use convert_case::{Case, Casing};
use fs::Fs;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use gpui::{AnyView, App, AsyncApp, Context, Entity, SharedString, Task, Window};
use language_model::{
    ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, LanguageModelToolChoice, LanguageModelToolSchemaFormat, RateLimiter,
};
use menu;
use open_ai::{
    ResponseStreamEvent,
    responses::{
        Request as ResponseRequest, StreamEvent as ResponsesStreamEvent,
        stream_response_with_headers,
    },
    stream_completion_with_headers,
};
use settings::{Settings, SettingsStore, update_settings_file};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};
use std::time::{Duration, Instant};
use ui::{ElevationIndex, Tooltip, prelude::*};
use ui_input::InputField;
use util::ResultExt;

use crate::provider::open_ai::{
    OpenAiEventMapper, OpenAiResponseEventMapper, into_open_ai, into_open_ai_response,
};
use collections::HashMap;
use http_client::{
    HttpClient,
    http::{HeaderMap, HeaderName, HeaderValue},
};

pub use settings::HeaderValueContent;
pub use settings::OpenAiCompatibleAvailableModel as AvailableModel;
pub use settings::OpenAiCompatibleModelCapabilities as ModelCapabilities;

const CONSECUTIVE_403_WARN_THRESHOLD: u32 = 3;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OpenAiCompatibleSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub headers: Option<HashMap<Arc<str>, HeaderValueContent>>,
    pub api_key_helper: Option<String>,
    pub api_key_helper_ttl_ms: Option<u64>,
}

pub struct OpenAiCompatibleLanguageModelProvider {
    id: LanguageModelProviderId,
    name: LanguageModelProviderName,
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

struct HelperKeyCache {
    key: Arc<str>,
    fetched_at: Instant,
}

struct HeaderEditor {
    name: Entity<InputField>,
    value: Entity<InputField>,
    is_env: bool,
}

pub struct State {
    id: Arc<str>,
    api_key_state: ApiKeyState,
    settings: OpenAiCompatibleSettings,
    helper_key_cache: Option<HelperKeyCache>,
    helper_key_invalidated: Arc<AtomicBool>,
    consecutive_403_count: Arc<AtomicU32>,
    helper_warning: Option<String>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        if self.settings.api_key_helper.is_some() && !self.is_env_var_conflict() {
            self.helper_key_cache.is_some()
        } else {
            self.api_key_state.has_key()
        }
    }

    fn is_env_var_conflict(&self) -> bool {
        let env_var_name = format!("{}_API_KEY", self.id).to_case(Case::UpperSnake);
        std::env::var(&env_var_name).is_ok_and(|v| !v.is_empty())
    }

    fn get_api_key(&self) -> Option<Arc<str>> {
        if self.settings.api_key_helper.is_some() && !self.is_env_var_conflict() {
            self.helper_key_cache.as_ref().map(|c| c.key.clone())
        } else {
            self.api_key_state.key(&self.settings.api_url)
        }
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let api_url = SharedString::new(self.settings.api_url.as_str());
        self.api_key_state
            .store(api_url, api_key, |this| &mut this.api_key_state, cx)
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if let Some(helper) = self.settings.api_key_helper.clone() {
            let env_var_name = format!("{}_API_KEY", self.id).to_case(Case::UpperSnake);

            if let Ok(env_value) = std::env::var(&env_var_name) {
                if !env_value.is_empty() {
                    self.helper_warning = Some(format!(
                        "The {} environment variable is already set. \
                         Remove it from your environment to use api_key_helper, \
                         otherwise it will be used instead.",
                        env_var_name
                    ));
                    cx.notify();
                    let api_url = SharedString::new(self.settings.api_url.clone());
                    return self
                        .api_key_state
                        .load_if_needed(api_url, |this| &mut this.api_key_state, cx);
                }
            }

            let ttl = self
                .settings
                .api_key_helper_ttl_ms
                .map(Duration::from_millis)
                .unwrap_or(Duration::MAX);

            let invalidated = self.helper_key_invalidated.load(Ordering::Relaxed);
            if !invalidated {
                if let Some(cache) = &self.helper_key_cache {
                    if cache.fetched_at.elapsed() < ttl {
                        return Task::ready(Ok(()));
                    }
                }
            }

            self.helper_key_invalidated.store(false, Ordering::Relaxed);
            return cx.spawn(async move |this, cx| {
                let helper_result = run_api_key_helper(&helper).await;
                this.update(cx, move |state, cx| {
                    match helper_result {
                        Ok(key) => {
                            state.helper_key_cache = Some(HelperKeyCache {
                                key,
                                fetched_at: Instant::now(),
                            });
                            state.helper_warning = None;
                            state.consecutive_403_count.store(0, Ordering::Relaxed);
                            cx.notify();
                            Ok(())
                        }
                        Err(err) => {
                            state.helper_key_cache = None;
                            state.helper_warning = Some(err.to_string());
                            cx.notify();
                            Err(AuthenticateError::Other(err))
                        }
                    }
                })
                .unwrap_or(Err(AuthenticateError::CredentialsNotFound))
            });
        }

        let api_url = SharedString::new(self.settings.api_url.clone());
        self.api_key_state
            .load_if_needed(api_url, |this| &mut this.api_key_state, cx)
    }
}

async fn run_api_key_helper(helper: &str) -> Result<Arc<str>> {
    let mut parts = helper.split_whitespace();
    let command = parts
        .next()
        .ok_or_else(|| anyhow!("api_key_helper path is empty"))?;
    let args: Vec<&str> = parts.collect();

    let output = util::command::new_command(command)
        .args(&args)
        .output()
        .await
        .map_err(|e| anyhow!("Failed to run api_key_helper '{}': {}", helper, e))?;

    if output.status.success() {
        let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if key.is_empty() {
            return Err(anyhow!(
                "api_key_helper '{}' produced empty output",
                helper
            ));
        }
        Ok(key.into())
    } else {
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output
            .status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let mut msg = format!(
            "api_key_helper '{}' failed (exit {})",
            helper, exit_code
        );
        if !stdout.trim().is_empty() {
            msg.push_str(&format!("\nstdout: {}", stdout.trim()));
        }
        if !stderr.trim().is_empty() {
            msg.push_str(&format!("\nstderr: {}", stderr.trim()));
        }
        Err(anyhow!(msg))
    }
}

impl OpenAiCompatibleLanguageModelProvider {
    pub fn new(id: Arc<str>, http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        fn resolve_settings<'a>(id: &'a str, cx: &'a App) -> Option<&'a OpenAiCompatibleSettings> {
            crate::AllLanguageModelSettings::get_global(cx)
                .openai_compatible
                .get(id)
        }

        let api_key_env_var_name = format!("{}_API_KEY", id).to_case(Case::UpperSnake).into();
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let Some(settings) = resolve_settings(&this.id, cx).cloned() else {
                    return;
                };
                if &this.settings != &settings {
                    let api_url = SharedString::new(settings.api_url.as_str());
                    this.api_key_state.handle_url_change(
                        api_url,
                        |this| &mut this.api_key_state,
                        cx,
                    );
                    if settings.api_key_helper != this.settings.api_key_helper
                        || settings.api_url != this.settings.api_url
                    {
                        this.helper_key_cache = None;
                        this.helper_key_invalidated.store(true, Ordering::Relaxed);
                    }
                    this.settings = settings;
                    cx.notify();
                }
            })
            .detach();
            let settings = resolve_settings(&id, cx).cloned().unwrap_or_default();
            State {
                id: id.clone(),
                api_key_state: ApiKeyState::new(
                    SharedString::new(settings.api_url.as_str()),
                    EnvVar::new(api_key_env_var_name),
                ),
                settings,
                helper_key_cache: None,
                helper_key_invalidated: Arc::new(AtomicBool::new(false)),
                consecutive_403_count: Arc::new(AtomicU32::new(0)),
                helper_warning: None,
            }
        });

        Self {
            id: id.clone().into(),
            name: id.into(),
            http_client,
            state,
        }
    }

    fn create_language_model(&self, model: AvailableModel) -> Arc<dyn LanguageModel> {
        Arc::new(OpenAiCompatibleLanguageModel {
            id: LanguageModelId::from(model.name.clone()),
            provider_id: self.id.clone(),
            provider_name: self.name.clone(),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }
}

impl LanguageModelProviderState for OpenAiCompatibleLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenAiCompatibleLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelProviderName {
        self.name.clone()
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAiCompat)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .settings
            .available_models
            .first()
            .map(|model| self.create_language_model(model.clone()))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .settings
            .available_models
            .iter()
            .map(|model| self.create_language_model(model.clone()))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(
        &self,
        _target_agent: language_model::ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(None, cx))
    }
}

pub struct OpenAiCompatibleLanguageModel {
    id: LanguageModelId,
    provider_id: LanguageModelProviderId,
    provider_name: LanguageModelProviderName,
    model: AvailableModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl OpenAiCompatibleLanguageModel {
    fn resolve_headers(settings: &OpenAiCompatibleSettings) -> Option<HeaderMap> {
        let mut headers = HeaderMap::new();
        if let Some(custom_headers) = &settings.headers {
            for (name, value) in custom_headers {
                let header_name = HeaderName::from_bytes(name.as_bytes()).log_err()?;
                let header_value = match value {
                    HeaderValueContent::Plain(val) => Some(val.clone()),
                    HeaderValueContent::Env(env) => std::env::var(env).ok(),
                };

                if let Some(val) = header_value {
                    let header_value = HeaderValue::from_str(&val).log_err()?;
                    headers.insert(header_name, header_value);
                }
            }
        }

        if headers.is_empty() {
            None
        } else {
            Some(headers)
        }
    }

    fn stream_completion(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let (api_key, settings, helper_key_invalidated, consecutive_403_count) =
            self.state.read_with(cx, |state, _cx| {
                (
                    state.get_api_key(),
                    state.settings.clone(),
                    state.helper_key_invalidated.clone(),
                    state.consecutive_403_count.clone(),
                )
            });

        let provider = self.provider_name.clone();
        let future = self.request_limiter.stream(async move {
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey { provider });
            };
            let headers = Self::resolve_headers(&settings);
            let result = stream_completion_with_headers(
                http_client.as_ref(),
                provider.0.as_str(),
                &settings.api_url,
                &api_key,
                request,
                headers,
            )
            .await
            .map_err(LanguageModelCompletionError::from);

            if let Err(LanguageModelCompletionError::PermissionError { .. }) = &result {
                if settings.api_key_helper.is_some() {
                    helper_key_invalidated.store(true, Ordering::Relaxed);
                    consecutive_403_count.fetch_add(1, Ordering::Relaxed);
                }
            }

            result
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }

    fn stream_response(
        &self,
        request: ResponseRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponsesStreamEvent>>>>
    {
        let http_client = self.http_client.clone();
        let (api_key, settings, helper_key_invalidated, consecutive_403_count) =
            self.state.read_with(cx, |state, _cx| {
                (
                    state.get_api_key(),
                    state.settings.clone(),
                    state.helper_key_invalidated.clone(),
                    state.consecutive_403_count.clone(),
                )
            });

        let provider = self.provider_name.clone();
        let future = self.request_limiter.stream(async move {
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey { provider });
            };
            let headers = Self::resolve_headers(&settings);
            let result = stream_response_with_headers(
                http_client.as_ref(),
                provider.0.as_str(),
                &settings.api_url,
                &api_key,
                request,
                headers,
            )
            .await
            .map_err(LanguageModelCompletionError::from);

            if let Err(LanguageModelCompletionError::PermissionError { .. }) = &result {
                if settings.api_key_helper.is_some() {
                    helper_key_invalidated.store(true, Ordering::Relaxed);
                    consecutive_403_count.fetch_add(1, Ordering::Relaxed);
                }
            }

            result
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for OpenAiCompatibleLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(
            self.model
                .display_name
                .clone()
                .unwrap_or_else(|| self.model.name.clone()),
        )
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        self.provider_id.clone()
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        self.provider_name.clone()
    }

    fn supports_tools(&self) -> bool {
        self.model.capabilities.tools
    }

    fn tool_input_format(&self) -> LanguageModelToolSchemaFormat {
        LanguageModelToolSchemaFormat::JsonSchemaSubset
    }

    fn supports_images(&self) -> bool {
        self.model.capabilities.images
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto => self.model.capabilities.tools,
            LanguageModelToolChoice::Any => self.model.capabilities.tools,
            LanguageModelToolChoice::None => true,
        }
    }

    fn supports_split_token_display(&self) -> bool {
        true
    }

    fn telemetry_id(&self) -> String {
        format!("openai/{}", self.model.name)
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_tokens
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        let max_token_count = self.max_token_count();
        cx.background_spawn(async move {
            let messages = super::open_ai::collect_tiktoken_messages(request);
            let model = if max_token_count >= 100_000 {
                // If the max tokens is 100k or more, it is likely the o200k_base tokenizer from gpt4o
                "gpt-4o"
            } else {
                // Otherwise fallback to gpt-4, since only cl100k_base and o200k_base are
                // supported with this tiktoken method
                "gpt-4"
            };
            tiktoken_rs::num_tokens_from_messages(model, &messages).map(|tokens| tokens as u64)
        })
        .boxed()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        if self.model.capabilities.chat_completions {
            let request = into_open_ai(
                request,
                &self.model.name,
                self.model.capabilities.parallel_tool_calls,
                self.model.capabilities.prompt_cache_key,
                self.max_output_tokens(),
                None,
            );
            let completions = self.stream_completion(request, cx);
            async move {
                let mapper = OpenAiEventMapper::new();
                Ok(mapper.map_stream(completions.await?).boxed())
            }
            .boxed()
        } else {
            let request = into_open_ai_response(
                request,
                &self.model.name,
                self.model.capabilities.parallel_tool_calls,
                self.model.capabilities.prompt_cache_key,
                self.max_output_tokens(),
                None,
            );
            let completions = self.stream_response(request, cx);
            async move {
                let mapper = OpenAiResponseEventMapper::new();
                Ok(mapper.map_stream(completions.await?).boxed())
            }
            .boxed()
        }
    }
}

struct ConfigurationView {
    api_key_editor: Entity<InputField>,
    api_key_helper_editor: Entity<InputField>,
    api_key_helper_ttl_editor: Entity<InputField>,
    header_editors: Vec<HeaderEditor>,
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
    last_save_error: Option<SharedString>,
}

impl ConfigurationView {
    fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| {
            InputField::new(
                window,
                cx,
                "000000000000000000000000000000000000000000000000000",
            )
        });

        let (api_key_helper_text, api_key_helper_ttl_text, current_headers) = {
            let state = state.read(cx);
            (
                state.settings.api_key_helper.clone(),
                state.settings.api_key_helper_ttl_ms.map(|v| v.to_string()),
                state.settings.headers.clone(),
            )
        };

        let api_key_helper_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, "/path/to/helper-script [args...]")
                .label("API Key Helper (optional)");
            if let Some(text) = &api_key_helper_text {
                input.set_text(text, window, cx);
            }
            input
        });

        let api_key_helper_ttl_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, "e.g. 3600000 for 1 hour")
                .label("API Key Helper TTL in ms (optional)");
            if let Some(text) = &api_key_helper_ttl_text {
                input.set_text(text, window, cx);
            }
            input
        });

        let mut header_editors = Vec::new();
        if let Some(headers) = current_headers {
            for (name, value) in &headers {
                let (value_str, is_env) = match value {
                    HeaderValueContent::Plain(v) => (v.clone(), false),
                    HeaderValueContent::Env(v) => (v.clone(), true),
                };
                let name_str = name.to_string();
                let name_editor = cx.new(|cx| {
                    let input = InputField::new(window, cx, "X-Custom-Header");
                    input.set_text(&name_str, window, cx);
                    input
                });
                let placeholder = if is_env { "VAR_NAME" } else { "Value" };
                let value_editor = cx.new(|cx| {
                    let input = InputField::new(window, cx, placeholder);
                    input.set_text(&value_str, window, cx);
                    input
                });
                header_editors.push(HeaderEditor {
                    name: name_editor,
                    value: value_editor,
                    is_env,
                });
            }
        }

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = Some(state.update(cx, |state, cx| state.authenticate(cx))) {
                    // We don't log an error, because "not signed in" is also an error.
                    let _ = task.await;
                }
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor,
            api_key_helper_editor,
            api_key_helper_ttl_editor,
            header_editors,
            state,
            load_credentials_task,
            last_save_error: None,
        }
    }

    fn add_header(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let name = cx.new(|cx| InputField::new(window, cx, "X-Custom-Header"));
        let value = cx.new(|cx| InputField::new(window, cx, "Value"));
        self.header_editors.push(HeaderEditor {
            name,
            value,
            is_env: false,
        });
        cx.notify();
    }

    fn remove_header(&mut self, index: usize, cx: &mut Context<Self>) {
        self.header_editors.remove(index);
        cx.notify();
    }

    fn save_settings(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let provider_id = self.state.read(cx).id.clone();

        let api_key_helper = {
            let text = self.api_key_helper_editor.read(cx).text(cx).trim().to_string();
            if text.is_empty() { None } else { Some(text) }
        };

        let api_key_helper_ttl_ms = {
            let raw = self.api_key_helper_ttl_editor.read(cx).text(cx).trim().to_string();
            if raw.is_empty() {
                None
            } else {
                match raw.parse::<u64>() {
                    Ok(ms) => Some(ms),
                    Err(_) => {
                        self.last_save_error = Some(
                            "API Key Helper TTL must be a positive integer (milliseconds)".into(),
                        );
                        cx.notify();
                        return;
                    }
                }
            }
        };

        let mut headers = HashMap::default();
        for editor in &self.header_editors {
            let name = editor.name.read(cx).text(cx).trim().to_string();
            let value = editor.value.read(cx).text(cx).trim().to_string();
            if name.is_empty() || value.is_empty() {
                continue;
            }
            let content = if editor.is_env {
                HeaderValueContent::Env(value)
            } else {
                HeaderValueContent::Plain(value)
            };
            headers.insert(Arc::<str>::from(name.as_str()), content);
        }

        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, move |settings, _cx| {
            if let Some(entry) = settings
                .language_models
                .get_or_insert_default()
                .openai_compatible
                .get_or_insert_default()
                .get_mut(&provider_id)
            {
                entry.api_key_helper = api_key_helper;
                entry.api_key_helper_ttl_ms = api_key_helper_ttl_ms;
                entry.headers = if headers.is_empty() { None } else { Some(headers) };
            }
        });

        self.last_save_error = None;
        cx.notify();
    }

    fn render_edit_section(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        v_flex()
            .gap_2()
            .mt_4()
            .child(
                Label::new("Provider Settings")
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            )
            .child(self.api_key_helper_editor.clone())
            .child(self.api_key_helper_ttl_editor.clone())
            .child(self.render_headers_section(cx))
            .when_some(self.last_save_error.clone(), |this, error| {
                this.child(
                    Label::new(error)
                        .color(Color::Error)
                        .size(LabelSize::Small),
                )
            })
            .child(
                Button::new("save-provider-settings", "Save")
                    .style(ButtonStyle::Outlined)
                    .full_width()
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.save_settings(window, cx);
                    })),
            )
    }

    fn render_headers_section(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        v_flex()
            .gap_2()
            .child(
                h_flex()
                    .justify_between()
                    .child(Label::new("Headers").size(LabelSize::Small))
                    .child(
                        Button::new("add-header", "Add Header")
                            .icon(IconName::Plus)
                            .icon_position(IconPosition::Start)
                            .icon_size(IconSize::XSmall)
                            .icon_color(Color::Muted)
                            .label_size(LabelSize::Small)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.add_header(window, cx);
                            })),
                    ),
            )
            .children((0..self.header_editors.len()).map(|ix| self.render_header_editor(ix, cx)))
    }

    fn render_header_editor(&self, ix: usize, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let header = &self.header_editors[ix];
        let (icon, tooltip) = if header.is_env {
            (IconName::Cog, "Environment variable name")
        } else {
            (IconName::Quote, "Plain text value")
        };

        h_flex()
            .gap_2()
            .items_center()
            .child(div().flex_1().child(header.name.clone()))
            .child(div().flex_1().child(header.value.clone()))
            .child(
                IconButton::new(("header-type", ix), icon)
                    .icon_size(IconSize::XSmall)
                    .tooltip(move |window, cx| Tooltip::text(tooltip)(window, cx))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        let header = &mut this.header_editors[ix];
                        header.is_env = !header.is_env;
                        let placeholder = if header.is_env { "VAR_NAME" } else { "Value" };
                        header.value.update(cx, |input, cx| {
                            input.set_placeholder_text(placeholder, window, cx);
                        });
                        cx.notify();
                    })),
            )
            .child(
                IconButton::new(("remove-header", ix), IconName::Trash)
                    .icon_size(IconSize::Small)
                    .icon_color(Color::Muted)
                    .on_click(cx.listener(move |this, _, _window, cx| {
                        this.remove_header(ix, cx);
                    })),
            )
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        // url changes can cause the editor to be displayed again
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn should_render_editor(&self, cx: &Context<Self>) -> bool {
        let state = self.state.read(cx);
        if state.settings.api_key_helper.is_some() && !state.is_env_var_conflict() {
            return false;
        }
        !state.is_authenticated()
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let env_var_set = state.api_key_state.is_from_env_var();
        let env_var_name = state.api_key_state.env_var_name();
        let has_helper = state.settings.api_key_helper.is_some();
        let env_var_conflict = state.is_env_var_conflict();
        let helper_warning = state.helper_warning.clone();
        let consecutive_403_count = state
            .consecutive_403_count
            .load(Ordering::Relaxed);

        let headers_section = state.settings.headers.as_ref().and_then(|headers| {
            let mut missing_envs = Vec::new();
            for value in headers.values() {
                match value {
                    HeaderValueContent::Plain(_) => {}
                    HeaderValueContent::Env(env) => {
                        if std::env::var(env).is_err() {
                            missing_envs.push(env.clone());
                        }
                    }
                }
            }

            if missing_envs.is_empty() {
                return None;
            }

            Some(
                v_flex().gap_1().mt_2().child(
                    Label::new(format!(
                        "Missing environment variables for headers: {}",
                        missing_envs.join(", ")
                    ))
                    .color(Color::Error)
                    .size(LabelSize::Small),
                ),
            )
        });

        let helper_warnings_section = if has_helper {
            let mut items: Vec<AnyElement> = Vec::new();

            if env_var_conflict {
                let env_name = format!("{}_API_KEY", state.id).to_case(Case::UpperSnake);
                items.push(
                    Label::new(format!(
                        "Warning: {} is set in your environment. \
                         Remove it to use api_key_helper.",
                        env_name
                    ))
                    .color(Color::Warning)
                    .size(LabelSize::Small)
                    .into_any_element(),
                );
            }

            if let Some(warning) = &helper_warning {
                items.push(
                    Label::new(warning.clone())
                        .color(Color::Error)
                        .size(LabelSize::Small)
                        .into_any_element(),
                );
            }

            if consecutive_403_count >= CONSECUTIVE_403_WARN_THRESHOLD {
                items.push(
                    Label::new(format!(
                        "Warning: Received {} consecutive authentication failures (403). \
                         Your api_key_helper may be returning an invalid key.",
                        consecutive_403_count
                    ))
                    .color(Color::Warning)
                    .size(LabelSize::Small)
                    .into_any_element(),
                );
            }

            if items.is_empty() {
                None
            } else {
                Some(v_flex().gap_1().mt_2().children(items))
            }
        } else {
            None
        };

        let api_key_section = if self.should_render_editor(cx) {
            v_flex()
                .on_action(cx.listener(Self::save_api_key))
                .child(Label::new("To use Zed's agent with an OpenAI-compatible provider, you need to add an API key."))
                .child(
                    div()
                        .pt(DynamicSpacing::Base04.rems(cx))
                        .child(self.api_key_editor.clone())
                )
                .child(
                    Label::new(
                        format!("You can also set the {env_var_name} environment variable and restart Zed."),
                    )
                    .size(LabelSize::Small).color(Color::Muted),
                )
                .when_some(headers_section, |this, section| this.child(section))
                .when_some(helper_warnings_section, |this, section| this.child(section))
                .into_any()
        } else if has_helper && !env_var_conflict {
            let helper_path = state.settings.api_key_helper.as_deref().unwrap_or("");
            v_flex()
                .child(
                    h_flex()
                        .mt_1()
                        .p_1()
                        .justify_between()
                        .rounded_md()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .bg(cx.theme().colors().background)
                        .child(
                            h_flex()
                                .flex_1()
                                .min_w_0()
                                .gap_1()
                                .child(Icon::new(IconName::Check).color(Color::Success))
                                .child(
                                    div()
                                        .w_full()
                                        .overflow_x_hidden()
                                        .text_ellipsis()
                                        .child(Label::new(format!(
                                            "API key via helper: {}",
                                            helper_path
                                        ))),
                                ),
                        ),
                )
                .when_some(headers_section, |this, section| this.child(section))
                .when_some(helper_warnings_section, |this, section| this.child(section))
                .into_any()
        } else {
            v_flex()
                .child(
                    h_flex()
                        .mt_1()
                        .p_1()
                        .justify_between()
                        .rounded_md()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .bg(cx.theme().colors().background)
                        .child(
                            h_flex()
                                .flex_1()
                                .min_w_0()
                                .gap_1()
                                .child(Icon::new(IconName::Check).color(Color::Success))
                                .child(
                                    div()
                                        .w_full()
                                        .overflow_x_hidden()
                                        .text_ellipsis()
                                        .child(Label::new(
                                            if env_var_set {
                                                format!("API key set in {env_var_name} environment variable")
                                            } else {
                                                format!("API key configured for {}", &state.settings.api_url)
                                            }
                                        ))
                                ),
                        )
                        .child(
                            h_flex()
                                .flex_shrink_0()
                                .child(
                                    Button::new("reset-api-key", "Reset API Key")
                                        .label_size(LabelSize::Small)
                                        .icon(IconName::Undo)
                                        .icon_size(IconSize::Small)
                                        .icon_position(IconPosition::Start)
                                        .layer(ElevationIndex::ModalSurface)
                                        .when(env_var_set, |this| {
                                            this.tooltip(Tooltip::text(format!("To reset your API key, unset the {env_var_name} environment variable.")))
                                        })
                                        .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx))),
                                ),
                        )
                )
                .when_some(headers_section, |this, section| this.child(section))
                .when_some(helper_warnings_section, |this, section| this.child(section))
                .into_any()
        };

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials…")).into_any()
        } else {
            v_flex()
                .size_full()
                .child(api_key_section)
                .child(self.render_edit_section(cx))
                .into_any()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;

    #[gpui::test]
    async fn test_resolve_headers_static(cx: &mut TestAppContext) {
        let mut custom_headers = HashMap::default();
        custom_headers.insert(
            "X-Custom-Static".into(),
            HeaderValueContent::Plain("static-value".into()),
        );

        let settings = OpenAiCompatibleSettings {
            api_url: "http://example.com".into(),
            available_models: vec![],
            headers: Some(custom_headers),
            api_key_helper: None,
            api_key_helper_ttl_ms: None,
        };

        let provider_id = LanguageModelProviderId::new("test");
        let http_client = http_client::FakeHttpClient::with_404_response();
        let _model = OpenAiCompatibleLanguageModel {
            id: LanguageModelId::from("test-model".to_string()),
            provider_id,
            provider_name: LanguageModelProviderName::new("test-provider"),
            model: AvailableModel {
                name: "test-model".into(),
                display_name: None,
                max_tokens: 100,
                max_output_tokens: None,
                max_completion_tokens: None,
                capabilities: ModelCapabilities::default(),
            },
            state: cx.new(|_| State {
                id: "test".into(),
                api_key_state: ApiKeyState::new(
                    "http://example.com".into(),
                    EnvVar::new("TEST_API_KEY".into()),
                ),
                settings: settings.clone(),
                helper_key_cache: None,
                helper_key_invalidated: Arc::new(AtomicBool::new(false)),
                consecutive_403_count: Arc::new(AtomicU32::new(0)),
                helper_warning: None,
            }),
            http_client,
            request_limiter: RateLimiter::new(1),
        };

        let headers = OpenAiCompatibleLanguageModel::resolve_headers(&settings)
            .expect("headers should be resolved");
        assert_eq!(headers.get("X-Custom-Static").unwrap(), "static-value");
    }

    #[gpui::test]
    async fn test_resolve_headers_env(cx: &mut TestAppContext) {
        unsafe { std::env::set_var("TEST_HEADER_VAR", "env-value") };

        let mut custom_headers = HashMap::default();
        custom_headers.insert(
            "X-Custom-Env".into(),
            HeaderValueContent::Env("TEST_HEADER_VAR".into()),
        );

        let settings = OpenAiCompatibleSettings {
            api_url: "http://example.com".into(),
            available_models: vec![],
            headers: Some(custom_headers),
            api_key_helper: None,
            api_key_helper_ttl_ms: None,
        };

        let provider_id = LanguageModelProviderId::new("test");
        let http_client = http_client::FakeHttpClient::with_404_response();
        let _model = OpenAiCompatibleLanguageModel {
            id: LanguageModelId::from("test-model".to_string()),
            provider_id,
            provider_name: LanguageModelProviderName::new("test-provider"),
            model: AvailableModel {
                name: "test-model".into(),
                display_name: None,
                max_tokens: 100,
                max_output_tokens: None,
                max_completion_tokens: None,
                capabilities: ModelCapabilities::default(),
            },
            state: cx.new(|_| State {
                id: "test".into(),
                api_key_state: ApiKeyState::new(
                    "http://example.com".into(),
                    EnvVar::new("TEST_API_KEY".into()),
                ),
                settings: settings.clone(),
                helper_key_cache: None,
                helper_key_invalidated: Arc::new(AtomicBool::new(false)),
                consecutive_403_count: Arc::new(AtomicU32::new(0)),
                helper_warning: None,
            }),
            http_client,
            request_limiter: RateLimiter::new(1),
        };

        let headers = OpenAiCompatibleLanguageModel::resolve_headers(&settings)
            .expect("headers should be resolved");
        assert_eq!(headers.get("X-Custom-Env").unwrap(), "env-value");

        unsafe { std::env::remove_var("TEST_HEADER_VAR") };
    }

    #[gpui::test]
    async fn test_resolve_headers_missing_env(cx: &mut TestAppContext) {
        unsafe { std::env::remove_var("MISSING_HEADER_VAR") };

        let mut custom_headers = HashMap::default();
        custom_headers.insert(
            "X-Custom-Env".into(),
            HeaderValueContent::Env("MISSING_HEADER_VAR".into()),
        );

        let settings = OpenAiCompatibleSettings {
            api_url: "http://example.com".into(),
            available_models: vec![],
            headers: Some(custom_headers),
            api_key_helper: None,
            api_key_helper_ttl_ms: None,
        };

        let provider_id = LanguageModelProviderId::new("test");
        let http_client = http_client::FakeHttpClient::with_404_response();
        let _model = OpenAiCompatibleLanguageModel {
            id: LanguageModelId::from("test-model".to_string()),
            provider_id,
            provider_name: LanguageModelProviderName::new("test-provider"),
            model: AvailableModel {
                name: "test-model".into(),
                display_name: None,
                max_tokens: 100,
                max_output_tokens: None,
                max_completion_tokens: None,
                capabilities: ModelCapabilities::default(),
            },
            state: cx.new(|_| State {
                id: "test".into(),
                api_key_state: ApiKeyState::new(
                    "http://example.com".into(),
                    EnvVar::new("TEST_API_KEY".into()),
                ),
                settings: settings.clone(),
                helper_key_cache: None,
                helper_key_invalidated: Arc::new(AtomicBool::new(false)),
                consecutive_403_count: Arc::new(AtomicU32::new(0)),
                helper_warning: None,
            }),
            http_client,
            request_limiter: RateLimiter::new(1),
        };

        let headers = OpenAiCompatibleLanguageModel::resolve_headers(&settings);
        assert!(
            headers.is_none(),
            "headers should not be resolved if env var is missing"
        );
    }
}

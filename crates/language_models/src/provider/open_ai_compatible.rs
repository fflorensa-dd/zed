use anyhow::{Result, anyhow};
use convert_case::{Case, Casing};
use futures::{FutureExt, StreamExt, future::BoxFuture};
use gpui::{AnyView, App, AsyncApp, Context, Entity, SharedString, Task, Window};
use http_client::HttpClient;
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
use settings::{Settings, SettingsStore};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use ui::{Banner, ElevationIndex, Severity, Tooltip, prelude::*};
use ui_input::InputField;
use util::ResultExt;

use crate::provider::open_ai::{
    OpenAiEventMapper, OpenAiResponseEventMapper, into_open_ai, into_open_ai_response,
};
pub use settings::OpenAiCompatibleAvailableModel as AvailableModel;
pub use settings::OpenAiCompatibleModelCapabilities as ModelCapabilities;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OpenAiCompatibleSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub api_key_helper: Option<String>,
    pub api_key_helper_ttl_secs: Option<u64>,
    pub custom_headers: Vec<(String, String)>,
}

/// Manages a dynamically-obtained API key from a helper command.
/// The cached key is stored in a shared Mutex so it can be updated from async contexts.
#[derive(Clone)]
struct ApiKeyHelper {
    command: String,
    ttl_secs: Option<u64>,
    // Shared so async tasks can update the cache without going through GPUI's entity system
    cached: Arc<Mutex<Option<(Arc<str>, Instant)>>>,
}

impl ApiKeyHelper {
    fn new(command: String, ttl_secs: Option<u64>) -> Self {
        Self {
            command,
            ttl_secs,
            cached: Arc::new(Mutex::new(None)),
        }
    }

    fn get_cached(&self) -> Option<Arc<str>> {
        let guard = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        let (key, fetched_at) = guard.as_ref()?;
        if let Some(ttl) = self.ttl_secs {
            if fetched_at.elapsed().as_secs() >= ttl {
                return None;
            }
        }
        Some(key.clone())
    }

    fn invalidate(&self) {
        if let Ok(mut guard) = self.cached.lock() {
            *guard = None;
        }
    }

    fn store(&self, key: Arc<str>) {
        if let Ok(mut guard) = self.cached.lock() {
            *guard = Some((key, Instant::now()));
        }
    }

    async fn fetch(command: String) -> Result<Arc<str>> {
        let output = smol::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .output()
            .await?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "api_key_helper exited with {}: {stderr}",
                output.status
            ));
        }
        let key = String::from_utf8(output.stdout)?;
        let key = key.trim();
        if key.is_empty() {
            return Err(anyhow!("api_key_helper produced empty output"));
        }
        Ok(key.into())
    }
}

pub struct OpenAiCompatibleLanguageModelProvider {
    id: LanguageModelProviderId,
    name: LanguageModelProviderName,
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    id: Arc<str>,
    api_key_state: ApiKeyState,
    settings: OpenAiCompatibleSettings,
    helper: Option<ApiKeyHelper>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key()
            || self
                .helper
                .as_ref()
                .and_then(|h| h.get_cached())
                .is_some()
    }

    /// Returns the effective API key with priority: env var > keychain > helper cache.
    fn effective_api_key(&self) -> Option<Arc<str>> {
        let api_url = &self.settings.api_url;
        if let Some(key) = self.api_key_state.key(api_url) {
            return Some(key);
        }
        self.helper.as_ref().and_then(|h| h.get_cached())
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let api_url = SharedString::new(self.settings.api_url.as_str());
        self.api_key_state
            .store(api_url, api_key, |this| &mut this.api_key_state, cx)
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let api_url = SharedString::new(self.settings.api_url.clone());
        let keychain_task = self
            .api_key_state
            .load_if_needed(api_url, |this| &mut this.api_key_state, cx);

        // If a helper is configured and no static key is available, run the helper too
        if self.settings.api_key_helper.is_some() && !self.api_key_state.has_key() {
            let helper = self.helper.clone();
            cx.spawn(async move |this, cx| {
                let _ = keychain_task.await;

                let still_no_static_key = this
                    .update(cx, |this, _| !this.api_key_state.has_key())
                    .unwrap_or(true);

                if still_no_static_key {
                    if let Some(helper) = helper {
                        if helper.get_cached().is_none() {
                            match ApiKeyHelper::fetch(helper.command.clone()).await {
                                Ok(key) => {
                                    helper.store(key);
                                    this.update(cx, |_, cx| cx.notify()).ok();
                                }
                                Err(err) => {
                                    log::error!("api_key_helper failed: {err}");
                                }
                            }
                        }
                    }
                }
                Ok(())
            })
        } else {
            keychain_task
        }
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
                    // Rebuild helper when settings change, preserving cache if command unchanged
                    this.helper = settings.api_key_helper.as_ref().map(|cmd| {
                        if let Some(old_helper) = &this.helper {
                            if old_helper.command == *cmd
                                && old_helper.ttl_secs == settings.api_key_helper_ttl_secs
                            {
                                // Reuse old helper with its cache intact
                                return old_helper.clone();
                            }
                        }
                        ApiKeyHelper::new(cmd.clone(), settings.api_key_helper_ttl_secs)
                    });
                    this.settings = settings;
                    cx.notify();
                }
            })
            .detach();
            let settings = resolve_settings(&id, cx).cloned().unwrap_or_default();
            let helper = settings
                .api_key_helper
                .as_ref()
                .map(|cmd| ApiKeyHelper::new(cmd.clone(), settings.api_key_helper_ttl_secs));
            State {
                id: id.clone(),
                api_key_state: ApiKeyState::new(
                    SharedString::new(settings.api_url.as_str()),
                    EnvVar::new(api_key_env_var_name),
                ),
                settings,
                helper,
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

        let (api_key, api_url, custom_headers, helper) =
            self.state.read_with(cx, |state, _cx| {
                (
                    state.effective_api_key(),
                    state.settings.api_url.clone(),
                    state.settings.custom_headers.clone(),
                    state.helper.clone(),
                )
            });

        let provider = self.provider_name.clone();
        let future = self.request_limiter.stream(async move {
            let api_key = match api_key {
                Some(key) => key,
                None => {
                    // No cached key yet — try the helper synchronously if available
                    if let Some(helper) = &helper {
                        match ApiKeyHelper::fetch(helper.command.clone()).await {
                            Ok(key) => {
                                helper.store(key.clone());
                                key
                            }
                            Err(err) => {
                                log::error!("api_key_helper failed: {err}");
                                return Err(LanguageModelCompletionError::NoApiKey { provider });
                            }
                        }
                    } else {
                        return Err(LanguageModelCompletionError::NoApiKey { provider });
                    }
                }
            };

            let result = stream_completion_with_headers(
                http_client.as_ref(),
                provider.0.as_str(),
                &api_url,
                &api_key,
                request.clone(),
                &custom_headers,
            )
            .await;

            match result {
                Ok(response) => Ok(response),
                Err(open_ai::RequestError::HttpResponseError { status_code, .. })
                    if status_code == http_client::http::StatusCode::UNAUTHORIZED =>
                {
                    // Token expired: refresh via helper and retry once
                    if let Some(helper) = &helper {
                        helper.invalidate();
                        match ApiKeyHelper::fetch(helper.command.clone()).await {
                            Ok(new_key) => {
                                helper.store(new_key.clone());
                                stream_completion_with_headers(
                                    http_client.as_ref(),
                                    provider.0.as_str(),
                                    &api_url,
                                    &new_key,
                                    request,
                                    &custom_headers,
                                )
                                .await
                                .map_err(Into::into)
                            }
                            Err(err) => {
                                log::error!("api_key_helper refresh failed: {err}");
                                Err(LanguageModelCompletionError::AuthenticationError {
                                    provider,
                                    message: "API key is invalid or expired".into(),
                                })
                            }
                        }
                    } else {
                        Err(LanguageModelCompletionError::AuthenticationError {
                            provider,
                            message: "Unauthorized".into(),
                        })
                    }
                }
                Err(e) => Err(e.into()),
            }
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

        let (api_key, api_url, custom_headers, helper) =
            self.state.read_with(cx, |state, _cx| {
                (
                    state.effective_api_key(),
                    state.settings.api_url.clone(),
                    state.settings.custom_headers.clone(),
                    state.helper.clone(),
                )
            });

        let provider = self.provider_name.clone();
        let future = self.request_limiter.stream(async move {
            let api_key = match api_key {
                Some(key) => key,
                None => {
                    if let Some(helper) = &helper {
                        match ApiKeyHelper::fetch(helper.command.clone()).await {
                            Ok(key) => {
                                helper.store(key.clone());
                                key
                            }
                            Err(err) => {
                                log::error!("api_key_helper failed: {err}");
                                return Err(LanguageModelCompletionError::NoApiKey { provider });
                            }
                        }
                    } else {
                        return Err(LanguageModelCompletionError::NoApiKey { provider });
                    }
                }
            };

            let result = stream_response_with_headers(
                http_client.as_ref(),
                provider.0.as_str(),
                &api_url,
                &api_key,
                request.clone(),
                &custom_headers,
            )
            .await;

            match result {
                Ok(response) => Ok(response),
                Err(open_ai::RequestError::HttpResponseError { status_code, .. })
                    if status_code == http_client::http::StatusCode::UNAUTHORIZED =>
                {
                    if let Some(helper) = &helper {
                        helper.invalidate();
                        match ApiKeyHelper::fetch(helper.command.clone()).await {
                            Ok(new_key) => {
                                helper.store(new_key.clone());
                                stream_response_with_headers(
                                    http_client.as_ref(),
                                    provider.0.as_str(),
                                    &api_url,
                                    &new_key,
                                    request,
                                    &custom_headers,
                                )
                                .await
                                .map_err(Into::into)
                            }
                            Err(err) => {
                                log::error!("api_key_helper refresh failed: {err}");
                                Err(LanguageModelCompletionError::AuthenticationError {
                                    provider,
                                    message: "API key is invalid or expired".into(),
                                })
                            }
                        }
                    } else {
                        Err(LanguageModelCompletionError::AuthenticationError {
                            provider,
                            message: "Unauthorized".into(),
                        })
                    }
                }
                Err(e) => Err(e.into()),
            }
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

    fn supports_streaming_tools(&self) -> bool {
        true
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
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
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
            state,
            load_credentials_task,
        }
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
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let env_var_set = state.api_key_state.is_from_env_var();
        let env_var_name = state.api_key_state.env_var_name().clone();
        let has_helper = state.settings.api_key_helper.is_some();

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
                .into_any()
        } else {
            v_flex()
                .gap_1()
                .when(env_var_set && has_helper, |this| {
                    this.child(
                        Banner::new()
                            .severity(Severity::Warning)
                            .child(Label::new(format!(
                                "Both {env_var_name} and api_key_helper are configured. \
                                 The environment variable takes precedence."
                            )).size(LabelSize::Small))
                    )
                })
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
                                            } else if has_helper {
                                                "API key obtained from helper command".to_string()
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
                                        .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                                        .layer(ElevationIndex::ModalSurface)
                                        .when(env_var_set, |this| {
                                            this.tooltip(Tooltip::text(format!("To reset your API key, unset the {env_var_name} environment variable.")))
                                        })
                                        .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx))),
                                ),
                        ),
                )
                .into_any()
        };

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials…")).into_any()
        } else {
            v_flex().size_full().child(api_key_section).into_any()
        }
    }
}

//! The one model client: Brama's OpenAI-compatible chat completions. The
//! caller's environment carries the gateway and its bearer (`BRAMA_URL`,
//! `BRAMA_API_KEY`) and, when the route requires a signed agent, the agent
//! identity (`WISENT_APP_AGENT_ID`, `WISENT_APP_AGENT_AUTH_SECRET`). No
//! provider key or provider host appears here: the Python agents reached
//! models through a router of their own and loaded local checkpoints with
//! transformers, which is inference outside Brama.

use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const URL: &str = "BRAMA_URL";
const KEY: &str = "BRAMA_API_KEY";
const AGENT: &str = "WISENT_APP_AGENT_ID";
const SECRET: &str = "WISENT_APP_AGENT_AUTH_SECRET";
const CHAT: &str = "/v1/chat/completions";

/// Generation settings the run declares. One left out is not sent, so the
/// route's own applies; the result records exactly these.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

pub struct Brama {
    url: String,
    api_key: String,
    agent: Option<(String, String)>,
}

fn setting(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

impl Brama {
    /// The gateway the environment names, or a refusal naming what is missing.
    pub fn from_env() -> Result<Self> {
        let url = setting(URL).ok_or_else(|| Error::Config(format!("{URL} is not set; a model seat needs Brama")))?;
        let api_key = setting(KEY).ok_or_else(|| Error::Config(format!("{KEY} is not set; a model seat needs Brama")))?;
        let agent = match (setting(AGENT), setting(SECRET)) {
            (Some(id), Some(secret)) => Some((id, secret)),
            (None, None) => None,
            _ => return Err(Error::Config(format!("{AGENT} and {SECRET} are set together or not at all"))),
        };
        Ok(Self {
            url: url.trim_end_matches('/').to_owned(),
            api_key,
            agent,
        })
    }

    /// `system` and `text` as the conversation to `route`; the answer's text.
    pub fn chat(&self, route: &str, system: &str, text: &str, sampling: &Sampling) -> Result<String> {
        let mut body = serde_json::to_value(sampling)
            .map_err(|error| Error::Model(format!("the request to {route} could not be written: {error}")))?;
        body["model"] = json!(route);
        body["messages"] = json!([
            { "role": "system", "content": system },
            { "role": "user", "content": text },
        ]);
        let body = body.to_string();
        let endpoint = format!("{}{CHAT}", self.url);
        let mut request = ureq::post(&endpoint)
            .set("content-type", "application/json")
            .set("authorization", &format!("Bearer {}", self.api_key));
        if let Some((id, secret)) = &self.agent {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| Error::Config(format!("the clock is before the epoch: {error}")))?
                .as_secs()
                .to_string();
            let digest = hex::encode(Sha256::digest(body.as_bytes()));
            let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
                .map_err(|error| Error::Config(format!("{SECRET} cannot key a signature: {error}")))?;
            mac.update(format!("{id}:{timestamp}:{digest}").as_bytes());
            let signature = hex::encode(mac.finalize().into_bytes());
            request = request
                .set("x-agent-id", id)
                .set("x-agent-timestamp", &timestamp)
                .set("x-agent-signature", &signature);
        }
        let raw = match request.send_string(&body) {
            Ok(response) => response
                .into_string()
                .map_err(|error| Error::Model(format!("{endpoint}: the answer of {route} could not be read: {error}")))?,
            Err(ureq::Error::Status(status, response)) => {
                let said = response.into_string().map_err(|error| {
                    Error::Model(format!("{endpoint} answered HTTP {status} for {route}, unreadably: {error}"))
                })?;
                return Err(Error::Model(format!("{endpoint} answered HTTP {status} for {route}: {said}")));
            }
            Err(error) => return Err(Error::Model(format!("{endpoint} could not be reached for {route}: {error}"))),
        };
        let answer: Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Model(format!("{endpoint} answered {route} with no JSON ({error}): {raw}")))?;
        answer["choices"]
            .as_array()
            .and_then(|choices| choices.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .filter(|content| !content.trim().is_empty())
            .map(str::to_owned)
            .ok_or_else(|| Error::Model(format!("{endpoint} answered {route} without message content: {raw}")))
    }
}

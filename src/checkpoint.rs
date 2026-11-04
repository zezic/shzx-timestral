//! Released checkpoints pinned to the revisions used by the reference fixtures.

#[cfg(feature = "hub")]
use std::path::PathBuf;

/// A released Timestral checkpoint, including its reproducible Hub revision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Model {
    /// The smallest checkpoint, suitable for quick local experiments.
    #[default]
    Timestral5M,
    Timestral28M,
    Timestral123M,
    Timestral209M,
}

impl Model {
    /// The official Hugging Face model repository.
    #[must_use]
    pub const fn repo_id(self) -> &'static str {
        match self {
            Self::Timestral5M => "mistralai/Timestral-5M",
            Self::Timestral28M => "mistralai/Timestral-28M",
            Self::Timestral123M => "mistralai/Timestral-123M",
            Self::Timestral209M => "mistralai/Timestral-209M",
        }
    }

    /// The immutable commit used by the backend's recorded reference outputs.
    ///
    /// Sources: the `nfnt/datafusion-forecast` golden workflow at commit
    /// `cd7bb7799fbe8f67dc61b400fb52185388e0c5db`.
    #[must_use]
    pub const fn revision(self) -> &'static str {
        match self {
            Self::Timestral5M => "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
            Self::Timestral28M => "49b26409ac87b5935539a5d8ffe7f397b720db47",
            Self::Timestral123M => "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
            Self::Timestral209M => "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
        }
    }

    /// Download the pinned configuration, weights and weight license to the Hub cache.
    ///
    /// Uses the usual Hugging Face token file or `HF_TOKEN`, and respects
    /// `HF_HOME` / `HF_HUB_CACHE`. Already cached files are reused. This blocks;
    /// call it outside a UI render thread. Use [`Self::cached`] to forbid network
    /// access, or load a local checkpoint directory without the `hub` feature.
    #[cfg(feature = "hub")]
    pub fn download(self) -> crate::Result<PathBuf> {
        self.resolve(false)
    }

    /// Resolve all three pinned files from the cache without network access.
    ///
    /// Returns an error when any required file is absent; never substitutes a
    /// different revision. This is the explicit offline counterpart to
    /// [`Self::download`].
    #[cfg(feature = "hub")]
    pub fn cached(self) -> crate::Result<PathBuf> {
        self.resolve(true)
    }

    #[cfg(feature = "hub")]
    fn resolve(self, local_files_only: bool) -> crate::Result<PathBuf> {
        let client = hf_hub::HFClientSync::new()?;
        let (owner, name) = hf_hub::split_id(self.repo_id());
        let repo = client.model(owner, name);
        let config = repo
            .download_file()
            .filename("config.json")
            .revision(self.revision())
            .local_files_only(local_files_only)
            .send()?;
        for filename in ["model.safetensors", "LICENSE"] {
            repo.download_file()
                .filename(filename)
                .revision(self.revision())
                .local_files_only(local_files_only)
                .send()?;
        }
        config
            .parent()
            .map(std::path::Path::to_path_buf)
            .ok_or_else(|| {
                crate::Error::InvalidInput("Hub configuration has no snapshot directory".into())
            })
    }
}

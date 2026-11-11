use super::RogueNetAgent;
use bevy::{
    asset::{io::Reader, Asset, AssetLoader, LoadContext},
    reflect::TypePath,
};

/// Bevy asset containing a [`RogueNetAgent`].
#[cfg_attr(docsrs, doc(cfg(feature = "bevy")))]
#[derive(Asset, TypePath)]
pub struct RogueNetAsset {
    pub agent: RogueNetAgent,
}

/// Bevy asset loader for loading .roguenet checkpoint archives as [`RogueNetAsset`]s.
#[cfg_attr(docsrs, doc(cfg(feature = "bevy")))]
#[derive(Default, TypePath)]
pub struct RogueNetAssetLoader;

impl AssetLoader for RogueNetAssetLoader {
    type Asset = RogueNetAsset;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<RogueNetAsset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let agent = RogueNetAgent::load_archive(std::io::Cursor::new(bytes))?;
        Ok(RogueNetAsset { agent })
    }

    fn extensions(&self) -> &[&str] {
        &["roguenet"]
    }
}

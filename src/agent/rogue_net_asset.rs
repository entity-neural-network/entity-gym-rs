use super::RogueNetAgent;
use anyhow::{anyhow, Error};
use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    reflect::TypeUuid,
    utils::BoxedFuture,
};

/// Bevy asset containing a [`RogueNetAgent`].
#[cfg_attr(docsrs, doc(cfg(feature = "bevy")))]
#[derive(TypeUuid)]
#[uuid = "5311f2d3-2c70-4490-bbfe-2f18cc885672"]
pub struct RogueNetAsset {
    pub agent: RogueNetAgent,
}

/// Bevy asset loader for loading .roguenet checkpoint archives as [`RogueNetAsset`]s.
#[cfg_attr(docsrs, doc(cfg(feature = "bevy")))]
#[derive(Default)]
pub struct RogueNetAssetLoader;

impl AssetLoader for RogueNetAssetLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            let cursor = std::io::Cursor::new(bytes);
            let agent =
                RogueNetAgent::load_archive(cursor).map_err(|e| anyhow!(format!("{}", e)))?;
            let asset = RogueNetAsset { agent };
            load_context.set_default_asset(LoadedAsset::new(asset));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["roguenet"]
    }
}

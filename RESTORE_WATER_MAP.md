# Restored water-level trend map

This package uses build **5.1.0**.

The missing map was the province-level trend map from the previous app. Build 5.0 only retained the basin map with exact-coordinate station markers. Build 5.1 restores the separate map with province shading and large rise/fall labels.

## GitHub update

Replace the contents of the new repository with this package, preserving the folders:

- `core/`
- `providers/`
- `data/`
- `.streamlit/`

Then in Streamlit Cloud:

1. Open **Manage app**.
2. Clear the cache.
3. Reboot the app.
4. Confirm that the sidebar shows `Clean package build 5.1.0`.
5. Keep **Show province water-level trend map** enabled.

The province map requires one of the configured Philippine government province-boundary services to be reachable. The app automatically tries a second source when the first fails.

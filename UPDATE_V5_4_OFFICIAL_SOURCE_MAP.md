# Build 5.4.0 update

## Minimum GitHub files to replace or add

```text
app.py
requirements.txt
core/target_river_map.py
providers/pagasa_bulletins.py
providers/llda_water_level.py
```

The `pypdf` line in `requirements.txt` is required because PAGASA may link current Flood Watch bulletins as PDF files.

After committing the files:

```text
Streamlit Cloud → Manage app → Clear cache → Reboot
```

Confirm that the sidebar shows:

```text
Clean package build 5.4.0
```

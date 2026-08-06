# Build 5.2 target-river correction

The earlier table searched `basin_name` together with river and station metadata. Because every PAGASA PMT row used the generic basin name `Pasig-Laguna`, those rows were also counted under Laguna. The table then selected the final row among stations sharing one observation timestamp, which could display the same level for more than one river group.

Build 5.2 uses strict river-system/province rules, shows a named reference station, and adds a dedicated map for Marikina, Tullahan, Meycauayan/MMORS, Pampanga, and Laguna. Large group markers are representative display anchors; only verified coordinates are plotted as exact station markers.

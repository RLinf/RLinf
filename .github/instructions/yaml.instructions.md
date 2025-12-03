---
description: 'YAML configuration file conventions and guidelines'
applyTo: '**/**.yaml'
---

# YAML Configuration File Conventions

## General Instructions

- Make sure newly-added configuration files' fields are aligned with existing configuration files in terms of fields and structure. Ignore optional fields that are not necessary for the new configuration.
- Pay attention to recent field modifications in existing configuration files and ensure consistency when adding similar YAML files.
- If a YAML configuration file adds new fields that are not present in existing files, ensure that all related configuration files are updated accordingly to maintain consistency.
- Use comments to explain non-obvious configuration fields or choices.
- Maintain a consistent indentation style (2 spaces per level) throughout the YAML file.
- Use lowercase letters and underscores for field names to maintain consistency.
- Use lowercase `true` and `false` for boolean values.
- DO NOT modify configuration fields in code that can be set by users in any circumstances. All fields should be treated as read-only.
# Method provenance

The hooked Qwen3 and Llama model implementations and hook-location conventions are migrated from the original PsySET source. MeanDiff, Probe, prompt steering, and PEFT SFT/DPO interfaces are retained in the multilingual package. BRIGHTER is used to construct emotion-label directions; it does not establish that emotion recognition directions control supportive response quality.

Model source files retain their upstream license headers. Their imports and package locations have changed, but their model internals are intentionally not rewritten during this migration.

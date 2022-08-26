<svelte:head>
	<title>IndieLabel</title>
</svelte:head>

<script lang="ts">
	import { onMount } from "svelte";
	import Section from "./Section.svelte";
	import IterativeClustering from "./IterativeClustering.svelte";
	import OverallResults from "./OverallResults.svelte";
	import Labeling from "./Labeling.svelte";
	import HypothesisPanel from "./HypothesisPanel.svelte"

	let personalized_model;
	let personalized_models = [];
	let breakdown_category;
	let breakdown_categories = [];
	let systems = ["Perspective comment toxicity classifier"]; // Only one system for now
	let clusters = [];
	let promise = Promise.resolve(null);

	function getAuditSettings() {
		fetch("./audit_settings")
			.then((r) => r.text())
			.then(function (r_orig) {
				let r = JSON.parse(r_orig);
				breakdown_categories = r["breakdown_categories"];
				breakdown_category = breakdown_categories[0];
				personalized_models = r["personalized_models"];
				personalized_model = personalized_models[0];
				clusters = r["clusters"];
			});
	}
	onMount(async () => {
		getAuditSettings();
	});

	function handleAuditButton() {
		promise = getAudit();
	}

	async function getAudit() {
		let req_params = {
			pers_model: personalized_model,
			breakdown_axis: breakdown_category,
			perf_metric: "avg_diff",
			breakdown_sort: "difference",
			n_topics: 10,
		};
		let params = new URLSearchParams(req_params).toString();
		const response = await fetch("./get_audit?" + params);
		const text = await response.text();
		const data = JSON.parse(text);
		return data;
	}

</script>

<main>
	<HypothesisPanel model={personalized_model} />

	<Labeling />

	<IterativeClustering clusters={clusters} ind={1} personalized_model={personalized_model} />

	<div id="audit-settings" class="section">
		<h5>Audit settings</h5>
		<Section
			section_id="systems"
			section_title="What status-quo system would you like to audit?"
			section_opts={systems}
			bind:value={systems[0]}
		/>
		<Section
			section_id="personalized_model"
			section_title="What model would you like to use to represent your views?"
			section_opts={personalized_models}
			bind:value={personalized_model}
		/>
		<Section
			section_id="breakdown_category"
			section_title="How would you like to explore the performance of the system?"
			section_opts={breakdown_categories}
			bind:value={breakdown_category}
		/>
		<button on:click={handleAuditButton}> Generate results </button>
		<div>
			Personalized model: {personalized_model}, Breakdown category: {breakdown_category}
		</div>
	</div>

	{#await promise}
		<p>...waiting</p>
	{:then audit_results}
		{#if audit_results}
			<OverallResults data={audit_results} clusters={clusters} personalized_model={personalized_model} />
		{/if}
	{:catch error}
		<p style="color: red">{error.message}</p>
	{/await}
</main>

<style>
	main {
		text-align: left;
		padding: 1em;
		max-width: 240px;
		margin: 0 0;
	}
	h3 {
		color: rgb(80, 80, 80);
		font-size: 30px;
	}
	h5 {
		color: rgb(80, 80, 80);
		font-size: 25px;
	}
	h6 {
		margin-top: 50px;
		text-transform: uppercase;
		font-size: 14px;
	}
	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>

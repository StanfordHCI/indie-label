<script lang="ts">
	import { onMount } from "svelte";
	import "../node_modules/svelte-material-ui/bare.css";

	import HypothesisPanel from "./HypothesisPanel.svelte";
	import MainPanel from "./MainPanel.svelte";
	import Explore from "./Explore.svelte";

	let personalized_model;
	let personalized_models = [];
	let error_type_options = ['Both', 'System is under-sensitive', 'System is over-sensitive', 'Show errors and non-errors'];
    let error_type = error_type_options[0];

	// Handle routing
	let searchParams = new URLSearchParams(window.location.search);
    let mode = searchParams.get("mode");
	let cur_user = searchParams.get("user");

	function getAuditSettings() {
		let req_params = {
            user: cur_user,
        };
		let params = new URLSearchParams(req_params).toString();
        fetch("./audit_settings?" + params)
            .then((r) => r.text())
            .then(function (r_orig) {
                let r = JSON.parse(r_orig);
				personalized_models = r["personalized_models"];
				personalized_model = personalized_models[0];
				cur_user = r["user"];
            });
	}
	onMount(async () => {
		getAuditSettings();
	});
</script>

<svelte:head>
	<title>IndieLabel</title>
</svelte:head>

<main>
	{#if mode == "explore"}
		<div>
			<Explore />
		</div>
	{:else }
		<div>
			{#key personalized_model }
				<HypothesisPanel model={personalized_model} cur_user={cur_user}/>
			{/key}

			<MainPanel bind:model={personalized_model} bind:error_type={error_type} cur_user={cur_user} on:change />
		</div>
	{/if}
</main>

<style>
</style>

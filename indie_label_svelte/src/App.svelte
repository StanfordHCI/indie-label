<script lang="ts">
	import { onMount } from "svelte";
	import "../node_modules/svelte-material-ui/bare.css";

	import HypothesisPanel from "./HypothesisPanel.svelte";
	import MainPanel from "./MainPanel.svelte";
	import SelectUserDialog from "./SelectUserDialog.svelte";
	import Explore from "./Explore.svelte";
	import Results from "./Results.svelte";
	import StudyLinks from "./StudyLinks.svelte";
	import { user } from './stores/cur_user_store.js';
	import { users } from "./stores/all_users_store.js";

	let personalized_model;
	let personalized_models = [];

	// let topic = "";

	let error_type_options = ['Both', 'System is under-sensitive', 'System is over-sensitive', 'Show errors and non-errors'];
    let error_type = error_type_options[0];

	// Handle routing
	let searchParams = new URLSearchParams(window.location.search);
    let mode = searchParams.get("mode");
	let cur_user = searchParams.get("user");

	// Set cur_user if it's provided in URL params
	if (cur_user !== null) {
		user.update((value) => cur_user);
	}

	// Handle user dialog
    let user_dialog_open = false;
    user.subscribe(value => {
		cur_user = value;
	});

	// Handle all users
	let all_users = [];
	async function getUsers() {    
		const response = await fetch("./get_users");
		const text = await response.text();
		const data = JSON.parse(text);
		all_users = data["users"];
		users.update((value) => all_users);
	}

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
				personalized_model = personalized_models[0]; // TEMP
				console.log("personalized_model", personalized_model);
				// personalized_model = "model_1632886687_iterA";
				// let clusters = r["clusters"];
                // topic = clusters[0]; // TEMP
            });

		// fetch("./audit_settings")
		// 	.then((r) => r.text())
		// 	.then(function (r_orig) {
		// 		let r = JSON.parse(r_orig);
		// 		personalized_models = r["personalized_models"];
		// 		personalized_model = personalized_models[0]; // TEMP
		// 		// personalized_model = "model_1632886687_iterA";
		// 		let clusters = r["clusters"];
        //         topic = clusters[0]; // TEMP
		// 	});
	}
	onMount(async () => {
		getAuditSettings();
		getUsers();
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
	{:else if mode == "results"}
	<div>
		<Results />
	</div>
	{:else if mode == "study_links"}
	<div>
		<StudyLinks />
	</div>
	{:else }
		<SelectUserDialog bind:open={user_dialog_open} cur_user={cur_user} />
		<div>
			{#key personalized_model }
				<HypothesisPanel model={personalized_model} bind:user_dialog_open={user_dialog_open}/>
			{/key}

			<MainPanel bind:model={personalized_model} bind:error_type={error_type} on:change />
		</div>
	{/if}
</main>

<style>
</style>

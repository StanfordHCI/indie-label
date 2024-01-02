<script lang="ts">
	import Labeling from "./Labeling.svelte";
	import Auditing from "./Auditing.svelte";

	import Tab, { Label } from "@smui/tab";
	import TabBar from "@smui/tab-bar";

	export let model;
	export let error_type;
	export let cur_user;

	// Handle routing
	let active = "labeling";
    let searchParams = new URLSearchParams(window.location.search);
    let tab = searchParams.get("tab");
	if (tab == "auditing") {
		active = "auditing";
	}
	
</script>

<svelte:head>
	<title>IndieLabel</title>
</svelte:head>

<div class="auditing_panel">
	<div class="tab_header">
		<TabBar tabs={["labeling", "auditing"]} let:tab bind:active>
			<Tab {tab}>
				<Label>{tab}</Label>
			</Tab>
		</TabBar>
	</div>

	<div class="panel_contents">
		<div>
			<div id="labeling" hidden={active == "auditing"} >
				<Labeling cur_user={cur_user}/>	
			</div>

			<div id="auditing" hidden={active == "labeling"} >
				<Auditing bind:personalized_model={model} bind:cur_error_type={error_type} cur_user={cur_user} on:change/>	
			</div>
		</div>

	</div>
</div>

<style>
	.panel_contents {
		padding: 50px;
		margin-top: 50px;
	}
</style>

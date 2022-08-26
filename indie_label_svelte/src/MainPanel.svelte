<script lang="ts">
	import Labeling from "./Labeling.svelte";
	import Auditing from "./Auditing.svelte";
	import AppOld from "./AppOld.svelte";

	import Tab, { Label } from "@smui/tab";
	import TabBar from "@smui/tab-bar";

	export let model;
	// export let topic;
	export let error_type;

	let app_versions = ["old", "new"];
	let app_version = "new";

	// Handle routing
	let active = "auditing";
    let searchParams = new URLSearchParams(window.location.search);
    let tab = searchParams.get("tab");
	if (tab == "labeling") {
		active = "labeling";
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
		<!-- VERSION SELECTION -->
		<!-- <div>
			<Section
				section_id="app_version"
				section_title="What app version do you want to use?"
				section_opts={app_versions}
                width_pct={40}
				bind:value={app_version}
			/>
		</div> -->

		{#if app_version == app_versions[0]}
			<!-- OLD VERSION -->
			<AppOld />
		{:else if app_version == app_versions[1]}
			<!-- NEW VERSION -->
			<div>
				<div id="labeling" hidden={active == "auditing"} >
					<Labeling/>	
				</div>

				<div id="auditing" hidden={active == "labeling"} >
					<Auditing bind:personalized_model={model} bind:cur_error_type={error_type} on:change/>	
				</div>
			</div>
		{/if}

		<!-- TEMP -->
		<!-- {#key model}
			<div>Model: {model}</div> 
		{/key} -->
	</div>
</div>

<style>
	.panel_contents {
		padding: 50px;
		margin-top: 50px;
	}
</style>

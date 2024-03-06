<script lang="ts">
	import Labeling from "./Labeling.svelte";
	import Auditing from "./Auditing.svelte";
	import About from "./About.svelte";

	import Tab, { Label } from "@smui/tab";
	import TabBar from "@smui/tab-bar";
	import { Icon } from "@smui/common";

	export let model;
	export let error_type;
	export let cur_user;

	// Handle routing
	let active = "labeling";
    let searchParams = new URLSearchParams(window.location.search);
    let tab = searchParams.get("tab");
	if (tab == "auditing") {
		active = "auditing";
	} else if (tab == "about") {
		active = "about";
	}
	
</script>

<svelte:head>
	<title>IndieLabel</title>
</svelte:head>

<div class="auditing_panel">
	<div class="tab_header">
		<TabBar tabs={["labeling", "auditing", "about"]} let:tab bind:active>
			<Tab {tab} minWidth={tab == "about"}>
				{#if tab == "about"}
					<Icon class="material-icons">info_outlined</Icon>
				{:else}
					<Label>{tab}</Label>
				{/if}
				
			</Tab>
		</TabBar>
	</div>

	<div class="panel_contents">
		<div>
			<div id="labeling" hidden={active != "labeling"} >
				<Labeling cur_user={cur_user}/>	
			</div>

			<div id="auditing" hidden={active != "auditing"} >
				<Auditing bind:personalized_model={model} bind:cur_error_type={error_type} cur_user={cur_user} on:change/>	
			</div>

			<div id="about" hidden={active != "about"} >
				<About />
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

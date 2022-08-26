<script lang="ts">
    import { onMount } from "svelte";
	import IterativeClustering from "./IterativeClustering.svelte";
	import Button, { Label } from "@smui/button";
	import Textfield from '@smui/textfield';
	import LinearProgress from "@smui/linear-progress";

    export let ind;
	export let hunch;
	export let model;
	export let topic;

	let example_block = false;
	let clusters;

	function getAuditSettings() {
		fetch("./audit_settings")
			.then((r) => r.text())
			.then(function (r_orig) {
				let r = JSON.parse(r_orig);
				clusters = r["clusters"];
			});
	}
	onMount(async () => {
		getAuditSettings();
	});

	function handleTestOnExamples() {
		example_block = true;
	}
</script>

<div>
	<div>
		<!-- <h6>Hunch {ind + 1}</h6> -->
		<h6>Topic:</h6>
		{topic}
	</div>
	<div class="spacing_vert">
		<h6>Your summary/suggestions:</h6>
		<Textfield
			style="width: 100%;"
			helperLine$style="width: 100%;"
			textarea
			bind:value={hunch}
			label="My current hunch is that..."
		>
		</Textfield>
		<!-- <Button
			on:click={handleTestOnExamples}
			class="button_float_right spacing_vert"
			variant="outlined"
		>
			<Label>Test on examples</Label>
		</Button> -->
	</div>

	<div class="spacing_vert">
        <Button on:click={null} variant="outlined">
            <Label>Save</Label>
        </Button>
        <Button on:click={null} variant="outlined">
            <Label>Submit</Label>
        </Button>
    </div>

	<!-- {#await example_block}
        <div class="app_loading">
            <LinearProgress indeterminate />
        </div>
    {:then} -->
		<!-- {#if example_block}
			<IterativeClustering clusters={clusters} ind={ind + 1} personalized_model={model} />
		{/if} -->
    <!-- {:catch error}
        <p style="color: red">{error.message}</p>
    {/await} -->
</div>

<style>
	/* * {
        z-index: 11;
        overflow-x: hidden;
    } */
</style>

<script lang="ts">
    import { onMount } from "svelte";
	import Button, { Label } from "@smui/button";
	import Textfield from '@smui/textfield';

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
	</div>

	<div class="spacing_vert">
        <Button on:click={null} variant="outlined">
            <Label>Save</Label>
        </Button>
        <Button on:click={null} variant="outlined">
            <Label>Submit</Label>
        </Button>
    </div>
</div>

<style>
</style>

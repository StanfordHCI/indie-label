<script>
    import { createEventDispatcher } from 'svelte';

    import Select, { Option } from "@smui/select";
    export let section_id;
    export let section_title;
    export let section_opts;
    export let value;
    export let width_pct;

    const dispatch = createEventDispatcher();

    function sendChange() {
        dispatch("select_changed", {});
    }
</script>

<div>
    <!-- <label for={section_id}>{section_title}</label> -->
    <Select 
        label={section_title} 
        bind:value id={section_id} 
        style="width: {width_pct}%"
        on:change
    >
        {#each section_opts as opt, i}
            <Option value={opt}>{opt}</Option>
        {/each}
    </Select>
</div>

<style>
    div {
        padding: 20px 10px;
    }
</style>

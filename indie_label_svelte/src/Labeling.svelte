<script lang="ts">
    import { onMount } from "svelte";
    import Section from "./Section.svelte";
    import TopicTraining from "./TopicTraining.svelte";
    import CommentTable from "./CommentTable.svelte";

    import Textfield from '@smui/textfield';
    import Button, { Label } from "@smui/button";
    import LinearProgress from '@smui/linear-progress';
    import Svelecte from '../node_modules/svelecte/src/Svelecte.svelte';

    export let cur_user;

    let model_name = "";
    let personalized_models = [];
    let existing_model_name;
    let label_modes = [
        "Create a new model",
        "Edit an existing model",
        // "Tune your model for a topic area",
        // "Set up a group-based model",
    ];
    
    let clusters_for_tuning = [];
    let topic;

    // Handle routing
    let label_mode = label_modes[0];
	let searchParams = new URLSearchParams(window.location.search);
    let req_label_mode = parseInt(searchParams.get("label_mode"));
    if (req_label_mode == 0) {
        label_mode = label_modes[0];
    } else if (req_label_mode == 1) {
        label_mode = label_modes[1];
    } else if (req_label_mode == 2) {
        // Unused; previous topic-based mode
        label_mode = label_modes[2];
    } else if (req_label_mode == 3) {
        // Unused; previous group-based mode
        label_mode = label_modes[3];
    } 

    // Handle group options
    let options_pol = ['Conservative', 'Liberal', 'Independent', 'Other'];
    let sel_pol;
    let options_relig = ['Not important', 'Not too important', 'Somewhat important', 'Very important'];
    let sel_relig;
    let options_race = ["White", "Black or African American", "Hispanic", "Asian", "American Indian or Alaskan Native", "Native Hawaiian or Pacific Islander", "Other"];
    let sel_race;
    let options_gender = ['Male', 'Female', 'Nonbinary'];
    let sel_gender;
    let options_lgbtq = ["Non-LGBTQ+", "LGBTQ+"];
    let sel_lgbtq;

    let options_axis = ["A: Political affiliation", "B: Gender", "C: Race", "D: LGBTQ+ Identity", "E: Importance of religion"]
    let selected_axis;

    let group_size;

    function getGroupSize() {
        let req_params = {
            sel_gender: sel_gender,
            sel_pol: sel_pol,
            sel_relig: sel_relig,
            sel_race: sel_race,
            sel_lgbtq: sel_lgbtq,
        };
        let params = new URLSearchParams(req_params).toString();
        fetch("./get_group_size?" + params)
			.then((r) => r.text())
			.then(function (r_orig) {
				let r = JSON.parse(r_orig);
                group_size = r["group_size"];
			});
    }

    let promise = Promise.resolve(null);
    function handleGroupModel() {
        promise = getGroupModel();
    }

    async function getGroupModel() {
        let req_params = {
            user: cur_user,
            model_name: model_name,
            sel_gender: sel_gender,
            sel_pol: sel_pol,
            sel_relig: sel_relig,
            sel_race: sel_race,
            sel_lgbtq: sel_lgbtq,
        };
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_group_model?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        return data
    }

    function getLabeling() {
        let req_params = {
            user: cur_user,
        };
        let params = new URLSearchParams(req_params).toString();
        fetch("./get_labeling?" + params)
			.then((r) => r.text())
			.then(function (r_orig) {
				let r = JSON.parse(r_orig);
                personalized_models = r["personalized_models"];
                model_name = r["model_name_suggestion"];
                clusters_for_tuning = r["clusters_for_tuning"];
                topic = clusters_for_tuning[0]["text"];
                existing_model_name = personalized_models[0];
			});
	}
	onMount(async () => {
		getLabeling();
	});
</script>
    
<div>
    <h3>Labeling</h3>

    <!-- MODE SELECTION -->
    {#if label_mode != label_modes[3]}
    <div id="audit-mode" class="section">
        <Section
            section_id="label_mode"
            section_title="What labeling mode do you want to use?"
            section_opts={label_modes}
            bind:value={label_mode}
            width_pct={40}
        />
    </div>
    {/if}

    {#if label_mode == label_modes[0]}
        <!-- NEW MODEL -->
        <div style="width: 80%">
            In this section, you’ll label some example comments to give a sense of your perspectives on what is toxic or not.
            We’ll then train a simple model (which we’ll refer to as "your model") that estimates what your toxicity rating would be for the full dataset (with tens of thousands of comments) based on an existing dataset of toxicity ratings provided by different users.
        </div>

        <div id="new-model" class="section">
            <h5>Create a New Model</h5>

            <Textfield 
                bind:value={model_name} 
                label="Name your personalized model"
                style="width: 40%"
            />

            <!-- Labeling -->
            <h6>Comments to label</h6>
            <ul>
                <li>
                    Comments with scores <b>0</b> and <b>1</b> will be allowed to <b>remain</b> on the platform.
                </li>
                <li>
                    Comments with scores <b>2</b>, <b>3</b>, or <b>4</b> will be <b>deleted</b> from the platform.
                </li>
                <li>
                    Given that some comments may lack context, if you're not sure, feel free to mark the <b>unsure</b> option to skip a comment.
                </li>
            </ul>

            <CommentTable mode={"train"} model_name={model_name} cur_user={cur_user}/>
        </div>
    {:else if label_mode == label_modes[1]}
        <!-- EXISTING MODEL -->
        <div id="existing-model" class="section">
            <h5>Edit an Existing Model</h5>
            {#key personalized_models}
            <Section
                section_id="personalized_model"
                section_title="Select Your Personalized Model"
                section_opts={personalized_models}
                bind:value={existing_model_name}
                width_pct={40}
            />
            {/key}

            <!-- Edit model -->
            <h6>Comments to label</h6>
            <ul>
                <li>
                    Comments with scores <b>0</b> and <b>1</b> will be allowed to <b>remain</b> on the platform.
                </li>
                <li>
                    Comments with scores <b>2</b>, <b>3</b>, or <b>4</b> will be <b>deleted</b> from the platform.
                </li>
                <li>
                    Given that some comments may lack context, if you're not sure, feel free to mark the <b>unsure</b> option to skip a comment.
                </li>
            </ul>
            {#key existing_model_name}
                <CommentTable mode={"view"} model_name={existing_model_name} cur_user={cur_user}/>
            {/key}
        </div>
    {:else if label_mode == label_modes[2]}
        <!-- Topic training -->
        <div class="audit_section">
            <div class="head_5">Topic model training</div>
            <p></p>
            <div class="section_indent">
                <div>
                    <p>In what topic area would you like to tune your model?</p>
                    <Svelecte 
                        options={clusters_for_tuning} 
                        labelAsValue={true}
                        bind:value={topic}
                        placeholder="Select topic"
                        on:change={null}
                        style="width: 50%"
                    >
                    </Svelecte>
                </div>

                <div style="padding-top: 30px">
                    <!-- Labeling -->
                    <h6>Comments to label</h6>
                    <ul>
                        <li>
                            Comments with scores <b>0</b> and <b>1</b> will be allowed to <b>remain</b> on the platform.
                        </li>
                        <li>
                            Comments with scores <b>2</b>, <b>3</b>, or <b>4</b> will be <b>deleted</b> from the platform.
                        </li>
                        <li>
                            Given that some comments may lack context, if you're not sure, feel free to mark the <b>unsure</b> option to skip a comment.
                        </li>
                    </ul>
                    {#key topic}
                    <TopicTraining topic={topic} model_name={model_name} cur_user={cur_user}/>
                    {/key}                    
                </div>

            </div>
        </div>
    {:else if label_mode == label_modes[3]}
        <!-- Group-based model setup -->
        <div class="head_5">Group model training</div>
        <p>Please select just <b>one</b> of these five demographic axes (A, B, C, D, or E) to identify with to set up your group-based model:</p>

        <div>
            <p><b>Demographic axes</b></p>
            <Svelecte 
                options={options_axis} 
                labelAsValue={true}
                bind:value={selected_axis}
                placeholder="Select demographic axis"
                on:change={null}
                style="width: 50%"
            >
            </Svelecte>
        </div>

        <div class="spacing_vert_40">
        <!-- {#if selected_axis != null}
        <p>For this axis, please select a group that you would like to identify with to set up your group-based model:</p>
        {/if} -->
        <div style="{selected_axis == options_axis[0] ? 'display:initial': 'display:none'}" >
            <p><b>A: Political affiliation</b></p>
            <Svelecte 
                options={options_pol} 
                labelAsValue={true}
                bind:value={sel_pol}
                placeholder="Select political affiliation"
                on:change={getGroupSize}
                style="width: 50%"
            >
            </Svelecte>
        </div>
        <!-- {:else if selected_axis == options_axis[1]} -->
        <div style="{selected_axis == options_axis[1] ? 'display:initial': 'display:none'}" >
            <p><b>B: Gender</b></p>
            <Svelecte 
                options={options_gender} 
                labelAsValue={true}
                bind:value={sel_gender}
                placeholder="Select gender"
                on:change={getGroupSize}
                style="width: 50%"
            >
            </Svelecte>
        </div>
        <!-- {:else if selected_axis == options_axis[2]} -->
        <div style="{selected_axis == options_axis[2] ? 'display:initial': 'display:none'}" >
            <p><b>C: Race (select all that apply)</b></p>
            <Svelecte 
                options={options_race} 
                labelAsValue={true}
                bind:value={sel_race}
                placeholder="Select race(s)"
                on:change={getGroupSize}
                style="width: 50%"
                multiple=true
            >
            </Svelecte>
        </div>
        <!-- {:else if selected_axis == options_axis[3]} -->
        <div style="{selected_axis == options_axis[3] ? 'display:initial': 'display:none'}" >
            <p><b>D: LGBTQ+ Identity</b></p>
            <Svelecte 
                options={options_lgbtq} 
                labelAsValue={true}
                bind:value={sel_lgbtq}
                placeholder="Select LGBTQ+ identity"
                on:change={getGroupSize}
                style="width: 50%"
            >
            </Svelecte>
        </div>
        <!-- {:else if selected_axis == options_axis[4]} -->
        <div style="{selected_axis == options_axis[4] ? 'display:initial': 'display:none'}" >
            <p><b>E: Importance of religion</b></p>
            <Svelecte 
                options={options_relig} 
                labelAsValue={true}
                bind:value={sel_relig}
                placeholder="Select importance of religion"
                on:change={getGroupSize}
                style="width: 50%"
            >
            </Svelecte>
        </div>
        <!-- {/if} -->
        </div>

        {#if group_size}
        <div class="spacing_vert_40">
            <b>Number of labelers with matching traits</b>: {group_size}
        </div>
        {/if}
        
        <div class=spacing_vert_60>
            <Button
                on:click={handleGroupModel}
                variant="outlined"
                class=""
                disabled={group_size == null}
            >
                <Label>Train group-based model</Label>
            </Button>
        </div>
        
        <!-- Performance -->
        {#await promise}
            <div class="app_loading spacing_vert_20">
                <LinearProgress indeterminate />
            </div>
        {:then group_model_res}
            {#if group_model_res}
            <div class="spacing_vert_20">
                <p>Model for your selected group memberships has been successfully tuned.</p>
                <p>MAE: {group_model_res["mae"]}</p>
            </div>
            {/if}
        {:catch error}
            <p style="color: red">{error.message}</p>
        {/await}
    {/if}
</div>

<style>
</style>
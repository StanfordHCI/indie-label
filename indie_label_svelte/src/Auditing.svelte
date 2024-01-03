<script lang="ts">
    import { onMount } from "svelte";
    import Section from "./Section.svelte";
    import KeywordSearch from "./KeywordSearch.svelte";
    import OverallResults from "./OverallResults.svelte";
    import ClusterResults from "./ClusterResults.svelte";
    import HelpTooltip from "./HelpTooltip.svelte";
    import TopicTraining from "./TopicTraining.svelte";

    import { error_type } from './stores/error_type_store.js';
    import { topic_chosen } from './stores/cur_topic_store.js';
    import { model_chosen } from './stores/cur_model_store.js';

    import Button, { Label } from "@smui/button";
    import LinearProgress from "@smui/linear-progress";
    import LayoutGrid, { Cell } from "@smui/layout-grid";
    import Radio from '@smui/radio';
    import FormField from '@smui/form-field';
    import{ Wrapper } from '@smui/tooltip';
    import IconButton from '@smui/icon-button';
    import Svelecte from '../node_modules/svelecte/src/Svelecte.svelte';

    export let personalized_model;
    export let cur_error_type = "Both";
    export let cur_user;

    let evidence = [];
    let show_audit_settings = false;

    let error_type_options = [
        {
            "opt": 'Both', 
            "descr": '(System is under- or over-sensitive)', 
            "help": "View both types of potential system errors"
        },
        {
            "opt": 'System is under-sensitive', 
            "descr": '(Incorrectly rates as non-toxic)', 
            "help": "Focus on system errors where the system labeled content as Non-toxic when it should have been labeled as Toxic."
        },
        {
            "opt": 'System is over-sensitive', 
            "descr": '(Incorrectly rates as toxic)', 
            "help": "Focus on system errors where the system labeled content as Toxic when it should have been labeled as Non-toxic."
        },
        {
            "opt": 'Show errors and non-errors', 
            "descr": '', 
            "help": "Also show cases that are not likely to be potential errors"
        },
    ]
    
    let personalized_models = [];
    let systems = ["YouSocial comment toxicity classifier"]; // Only one system for now
    let clusters = [];
    let clusters_for_tuning = []
    let promise = Promise.resolve(null);

    // Handle routing
    let searchParams = new URLSearchParams(window.location.search);
    let scaffold_method = searchParams.get("scaffold");
    let mode = searchParams.get("mode");
    let topic_vis_method = searchParams.get("topic_vis_method");

    // Set audit type
    let audit_types = [
        "All topic exploration", 
        "Single topic exploration"
    ];
    let audit_type;
    if (scaffold_method == "fixed" || scaffold_method == "personal" || scaffold_method == "personal_group" || scaffold_method == "personal_test" || scaffold_method == "personal_cluster" || scaffold_method == "topic_train" || scaffold_method == "prompts") {
        audit_type = audit_types[1];
    } else {
        // No scaffolding mode or tutorial
        audit_type = audit_types[0];
    }

    let show_topic_training = false;
    if (scaffold_method == "topic_train") {
        show_topic_training = true;
    } 

    // Handle non-model mode
    let use_model = true;
    if (mode == "no_model") {
        use_model = false;
        cur_error_type = "Show errors and non-errors";
    }

    // Handle group model 
    let use_group_model = false;
    if (scaffold_method == "personal_group") {
        use_group_model = true;
    }
    
    let promise_cluster = Promise.resolve(null);

    // Get current topic from store
    let topic;
    topic_chosen.subscribe(value => {
        topic = value;
        handleClusterButton(); // re-render cluster results
	});

    // Get current model from store
    model_chosen.subscribe(value => {
        personalized_model = value;
        // Add to personalized_models if not there
        if (!personalized_models.includes(personalized_model)) {
            personalized_models.push(personalized_model);
        }
        getAuditResults();
	});

    // Save current error type
    async function updateErrorType() {    
		error_type.update((value) => cur_error_type);
        handleAuditButton();
        handleClusterButton();
	}

    async function updateTopicChosen() {
        if (topic != null) {
            topic_chosen.update((value) => topic);
        }
    }

    function getAuditResults() {
        let req_params = {
            user: cur_user,
            scaffold_method: scaffold_method,
        };
        let params = new URLSearchParams(req_params).toString();
        fetch("./audit_settings?" + params)
            .then((r) => r.text())
            .then(function (r_orig) {
                let r = JSON.parse(r_orig);
                personalized_models = r["personalized_models"];
                if (use_group_model) {
                    let personalized_model_grp = r["personalized_model_grp"];
                    personalized_model = personalized_model_grp[0];
                } else {
                    personalized_model = personalized_models[0];  // TEMP
                }
                
                model_chosen.update((value) => personalized_model);
                clusters = r["clusters"];
                clusters_for_tuning = r["clusters_for_tuning"];
                topic = clusters[0]["options"][0]["text"];
                topic_chosen.update((value) => topic);
                handleAuditButton(); 
                handleClusterButton();
            });
    }
    onMount(async () => {
        getAuditResults();
    });

    function handleAuditButton() {
        model_chosen.update((value) => personalized_model);
        if (personalized_model == "" || personalized_model == undefined) {
            return;
        }
        promise = getAudit(personalized_model);
    }

    async function getAudit(pers_model) {
        let req_params = {
            pers_model: pers_model,
            perf_metric: "avg_diff",
            breakdown_sort: "difference",
            n_topics: 10,
            error_type: "Both", // Only allow both error types
            cur_user: cur_user,
            topic_vis_method: topic_vis_method,
        };
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_audit?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        return data;
    }

    function handleClusterButton() {
		promise_cluster = getCluster(personalized_model);
	}

	async function getCluster(pers_model) {
        if (pers_model == "" || pers_model == undefined) {
            return null;
        }
		let req_params = {
			cluster: topic,
			topic_df_ids: [],
            cur_user: cur_user,
			pers_model: pers_model,
			example_sort: "descending", // TEMP
			comparison_group: "status_quo", // TEMP
			search_type: "cluster",
			keyword: "",
            error_type: cur_error_type,
            use_model: use_model,
            scaffold_method: scaffold_method,
		};
		let params = new URLSearchParams(req_params).toString();
		const response = await fetch("./get_cluster_results?" + params);
		const text = await response.text();
		const data = JSON.parse(text);
		return data;
	}
</script>

<div>
    <!-- 0: Audit settings -->
    <div>
        <div style="margin-top: 30px">
            <span class="head_3">Auditing</span>
        </div>
        <div style="width: 80%">
            {#if personalized_model}
            <p>In this section, we'll be auditing the content moderation system. Here, you’ll be aided by a personalized model that will help direct your attention towards potential problem areas in the model’s performance. This model isn’t meant to be perfect, but is designed to help you better focus on areas that need human review.</p>
            {:else}
            <p>Please first train your personalized model by following the steps in the "Labeling" tab (click the top left tab above).</p>
            {/if}
        </div>
        
        {#if show_audit_settings}
        <div class="audit_section">
            <div class="head_5">Audit settings</div>
            <div style="width: 50%">
                <p>Choose your audit settings here. These settings will affect all of the visualizations that follow, so you can return back here to make changes.</p>
            </div>
            <div class="section_indent">
                <Section
                    section_id="systems"
                    section_title="What status-quo system would you like to audit?"
                    section_opts={systems}
                    bind:value={systems[0]}
                    width_pct={40}
                />
                {#key personalized_model}
                    <Section
                        section_id="personalized_model"
                        section_title="What model would you like to use to represent your views?"
                        section_opts={personalized_models}
                        bind:value={personalized_model}
                        width_pct={40}
                        on:change
                    />
                {/key}

                <Section
                    section_id="audit_type"
                    section_title="What type of audit are you conducting?"
                    section_opts={audit_types}
                    bind:value={audit_type}
                    width_pct={40}
                    on:change
                />
            
                <LayoutGrid>
                    <Cell span={7}>
                        <Button
                            on:click={handleAuditButton}
                            variant="outlined"
                            class="button_float_right"
                        >
                            <Label>Start your audit</Label>
                        </Button>
                    </Cell>
                </LayoutGrid>
            </div>
        </div>
        {/if}
        {#if personalized_model}
        <p>Current model: {personalized_model}</p>
        {/if}
    </div>

    <!-- 1: All topics overview -->
    {#if personalized_model}
    {#if audit_type == audit_types[0]}
    <div class="audit_section">
        <div class="head_5">Overview of all topics</div>
        <p>First, browse the system performance by different auto-generated comment topic areas.</p>

        <div class="section_indent">            
            {#await promise}
                <div class="app_loading">
                    <LinearProgress indeterminate />
                </div>
            {:then audit_results}
                {#if audit_results}
                    <OverallResults
                        data={audit_results}
                        clusters={clusters}
                        personalized_model={personalized_model}
                        cluster={topic}
                    />
                {/if}
            {:catch error}
                <p style="color: red">{error.message}</p>
            {/await}
        </div>
    </div>
    {/if}

    <!-- 2a: Topic training -->
    {#if show_topic_training}
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
                <TopicTraining topic={topic} cur_user={cur_user}/>
                {/key}                    
            </div>

        </div>
    </div>
    {/if}

    <!-- 2: Topic overview -->
    <div class="audit_section">
        <div class="head_5">Topic exploration</div>
        <p></p>
        <div class="section_indent">
            <div>
                <div>
                    <p><b>What topic would you like to explore further?</b></p>
                    <Svelecte 
                        options={clusters} 
                        labelAsValue={true}
                        bind:value={topic}
                        placeholder="Select topic"
                        on:change={updateTopicChosen}
                        style="width: 50%"
                    >
                    </Svelecte>
                </div>

                {#if use_model}
                <div style="padding-top: 30px">
                    <p><b>What kind of system errors do you want to focus on?</b></p>
                    {#each error_type_options as e}
                        <div style="display: flex; align-items: center;">
                            <Wrapper rich>
                                <FormField>
                                    <Radio bind:group={cur_error_type} value={e.opt} on:change={updateErrorType} color="secondary" />
                                    <span slot="label">
                                        <b>{e.opt}</b> {e.descr}
                                        <IconButton class="material-icons" size="button" disabled>help_outline</IconButton>
                                    </span>
                                </FormField>
                                <HelpTooltip text={e.help} />
                            </Wrapper>
                        </div>
                    {/each}
                </div>
                {/if}
            </div>

            <div style="padding-top: 30px">
                {#await promise_cluster}
                    <div class="app_loading">
                        <LinearProgress indeterminate />
                    </div>
                {:then cluster_results}
                    {#if cluster_results}
                        {#if topic}
                        <ClusterResults 
                            cluster={topic} 
                            clusters={clusters} 
                            model={personalized_model} 
                            data={cluster_results} 
                            table_width_pct={100}
                            table_id={"main"}
                            use_model={use_model}
                            bind:evidence={evidence} 
                            on:change
                        />
                        {/if}
                    {/if}
                {:catch error}
                    <p style="color: red">{error.message}</p>
                {/await}
            </div>

        </div>
    </div>

    <!-- 3: Gather evidence -->
    <div class="audit_section">
        <div class="head_5">Gather additional evidence</div>
        <p>Next, you can optionally search for more comments to serve as evidence through manual keyword search (for individual words or phrases).</p>
        <div class="section_indent">
            {#key error_type}
            <KeywordSearch clusters={clusters} personalized_model={personalized_model} cur_user={cur_user} bind:evidence={evidence} use_model={use_model} on:change/>
            {/key}
        </div>
    </div>

    <!-- 4: Test hunch -->
    <div class="audit_section">
        <div class="head_5">Finalize your current report</div>
        <p>Finally, review the report you've generated on the side panel and provide a brief summary of the problem you see. You may also list suggestions or insights into addressing this problem if you have ideas. This report will be directly used by the model developers to address the issue you've raised</p>
    </div>
    {/if}
</div>

<style>
</style>
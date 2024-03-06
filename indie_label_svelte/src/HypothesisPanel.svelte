<script lang="ts">
    import { onMount } from "svelte";
    import { fade } from 'svelte/transition';
    import ClusterResults from "./ClusterResults.svelte";
    import SubmitReportDialog from "./SubmitReportDialog.svelte";

    import Button, { Label } from "@smui/button";
    import Textfield from '@smui/textfield';
    import Select, { Option } from "@smui/select";
    import { new_evidence } from './stores/new_evidence_store.js';
    import { open_evidence } from './stores/open_evidence_store.js';
    import { topic_chosen } from './stores/cur_topic_store.js';
    
    import Drawer, {
        AppContent,
        Content,
        Header,
        Title,
        Subtitle,
    } from '@smui/drawer';
    import List, { Item, Text, Graphic, PrimaryText, SecondaryText } from '@smui/list';
    import LinearProgress from "@smui/linear-progress";
    import CircularProgress from '@smui/circular-progress';
    import Checkbox from '@smui/checkbox';
    import FormField from '@smui/form-field';
    import IconButton from "@smui/icon-button";
    import { Icon } from "@smui/common";
    import Radio from '@smui/radio';

    export let model;
    export let cur_user;

    let all_reports = [];
    let cur_topic;
    topic_chosen.subscribe(value => {
		cur_topic = value;
	});
    // Handle submit report dialog
	let submit_dialog_open = false;

    // Handle routing
    let searchParams = new URLSearchParams(window.location.search);
    let scaffold_method = searchParams.get("scaffold");
    if (scaffold_method == null) {
        scaffold_method = "personal"; // Default to personalized model scaffold
    }
    let topic_vis_method = searchParams.get("topic_vis_method");

    // Handle drawer
    let open = false;
    let selected = null;
    let promise = Promise.resolve(null);
    let editTitle = false;
    let editErrorType = false;
    let unfinished_count = 0;
    let has_complete_report = false;
    let save_check_visible = false;  // Whether the save checkmark is visible
    
    function setActive(value: string) {
        selected = value;
        // Set local and store value of open evidence to selected report's
        cur_open_evidence = selected["evidence"];
        open_evidence.update((value) => cur_open_evidence);
        let isolated_topic = selected["title"].replace(/^(Topic: )/,'');

        // Close panel
        open = false;

        // Update topic if in personal mode
        if (scaffold_method == "personal" || scaffold_method == "personal_group" || scaffold_method == "personal_test" || scaffold_method == "tutorial") {
            topic_chosen.update((value) => isolated_topic);
        }
    }

    onMount(async () => {
        promise = getReports();
    });

    async function getReports() {
        if (model == "" || model == undefined){
            return [];
        }
        let req_params = {
            cur_user: cur_user,
            scaffold_method: scaffold_method,
            model: model,
            topic_vis_method: topic_vis_method,
        };
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_reports?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        all_reports = data["reports"]
        // Select first report initially
        selected = all_reports[0];
        setActive(selected);
        cur_open_evidence = selected["evidence"];
        unfinished_count = all_reports.filter(item => (item.evidence.length == 0) || (item.text_entry == "") || !(item.sep_selection)).length
        has_complete_report = hasCompleteReport();
        return all_reports;
    }

    // Handle evidence saving
    let cur_open_evidence = [];
    new_evidence.subscribe(value => {
        if (value != []) {
            // Check if any values with the same ID exist
            for (let i = 0; i < cur_open_evidence.length; i++) {
                if (cur_open_evidence[i]["id"] == value["id"]) {
                    return; // If so, don't add the item
                }
            }
            cur_open_evidence = cur_open_evidence.concat(value); // add new evidence item

            // Add to open evidence in store
            open_evidence.update((value) => cur_open_evidence);
            // Save to selected value
            if (selected != null) {
                selected["evidence"] = cur_open_evidence;
            }
        }
	});

    // Handle evidence removal
    open_evidence.subscribe(value => {
        if ((value != cur_open_evidence) && (value.length < cur_open_evidence.length)) {
            // Update local open evidence
            cur_open_evidence = value;
            // Save to selected value
            if (selected != null) {
                selected["evidence"] = cur_open_evidence;
            }
        }
	});

    let promise_save = Promise.resolve(null);
    function handleSaveReport() {
        // Briefly display checkmark
        promise_save = saveReport();
        save_check_visible = true;
        // Hide save checkmark after 1 second
        setTimeout(() => save_check_visible = false, 1000);
    }

    async function saveReport() {
        let req_params = {
            cur_user: cur_user,
            reports: JSON.stringify(all_reports),
            scaffold_method: scaffold_method,
            model: model,
        };
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./save_reports?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        
        has_complete_report = hasCompleteReport();
        return data;
    }

    function handleNewReport() {
        let new_report = {
            title: "",
            error_type: "",
            evidence: [],
            text_entry: "",
            sep_selection: "",
        };
        all_reports = all_reports.concat(new_report);
        promise = Promise.resolve(all_reports);
        // Open this new report
        selected = all_reports[all_reports.length - 1];
        cur_open_evidence = selected["evidence"];
        unfinished_count = all_reports.filter(item => (item.evidence.length == 0) || (item.text_entry == "") || !(item.sep_selection)).length
    }

    function handleDeleteReport() {
        // Remove selected item from reports
        all_reports = all_reports.filter(item => item != selected);
        promise = Promise.resolve(all_reports);
        selected  = all_reports[0];
        cur_open_evidence = selected["evidence"];
        unfinished_count = all_reports.filter(item => (item.evidence.length == 0) || (item.text_entry == "") || !(item.sep_selection)).length
    }

    function hasCompleteReport() {
        return all_reports.some(item => (item.evidence.length > 0) && (item.text_entry != "") && (item.sep_selection));
    }

    // Error type
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

    // Save current error type
    async function updateErrorType() {    
        // Update error type on main page to be the selected error type
        editErrorType = false;
	}

    // SEP selection
    let all_sep_options = [
        "Accuracy",
        "Bias/Discrimination",
        "Adversarial Example",
        "Other",
    ];

    let promise_submit = Promise.resolve(null);
    function handleSubmitReport() {
        promise_submit = submitReport();
    }
    async function submitReport() {
        submit_dialog_open = true;
        return true;
    }

</script>

<div>
    {#await promise_submit}
        <CircularProgress style="height: 32px; width: 32px;" indeterminate />
    {:then}
        <SubmitReportDialog bind:open={submit_dialog_open} cur_user={cur_user} all_reports={all_reports}/>
    {:catch error}
        <p style="color: red">{error.message}</p>
    {/await}
    <div class="hypothesis_panel">
        <div class="panel_header">
            <div class="panel_header_content">
                <div class="page_header">
                    <img src="/logo.png" style="height: 50px; padding: 0px 20px;" alt="IndieLabel" />
                    <Button class="user_button" color="secondary" style="margin: 12px 10px;" >
                        <Label>User: {cur_user}</Label>
                    </Button>
                </div>
                <div class="hypotheses_header">
                    <h5 style="float: left; margin: 0; padding: 5px 20px;">Your Audit Reports</h5>
                    <Button 
                        on:click={() => (open = !open)}
                        color="primary"
                        disabled={model == null}
                        style="float: right; padding: 10px; margin-right: 10px;"
                    >
                        {#if open}
                        <Label>Close</Label>
                        {:else}
                            {#key unfinished_count}
                            <Label>Unfinished reports ({unfinished_count})</Label>
                            {/key}
                        {/if}
                    </Button>
                </div>
            </div>
        </div>

        {#if model == null}
        <div class="panel_contents">
            <p>You can start to author audit reports in this panel after you've trained your personalized model in the "Labeling" tab.</p>
        </div>
        {:else}
        <div class="panel_contents">  
            <!-- Drawer -->
            {#await promise}
                <div class="app_loading_fullwidth">
                    <LinearProgress indeterminate />
                </div>
            {:then reports}
                {#if reports}
                <div class="drawer-container">
                    {#key open}
                    <Drawer variant="dismissible" bind:open>
                        <Header>
                            <Title>Your Reports</Title>
                            <Subtitle>Select a report to view.</Subtitle>
                        </Header>
                        <Content>
                            <List twoLine>
                                {#each reports as report}
                                    <Item
                                        href="javascript:void(0)"
                                        on:click={() => setActive(report)}
                                        activated={selected === report}
                                    >   
                                        {#if (report["evidence"].length > 0) && (report["text_entry"] != "") && (report["sep_selection"])}
                                        <Graphic class="material-icons" aria-hidden="true">task_alt</Graphic>
                                        {:else}
                                        <Graphic class="material-icons" aria-hidden="true">radio_button_unchecked</Graphic>
                                        {/if}
                                        <Text>
                                            <PrimaryText>
                                                {report["title"]}
                                            </PrimaryText>
                                            <SecondaryText>
                                                {report["error_type"]}
                                            </SecondaryText>
                                        </Text>
                                    </Item>
                                {/each}
                            </List>
                        </Content>
                    </Drawer>
                    {/key}
                    <AppContent class="app-content">
                        <main class="main-content">
                            {#if selected}
                            <div class="head_6_highlight">
                                Current Report
                            </div>
                            <div class="panel_contents2">
                                <!-- Title -->
                                <div class="spacing_vert">
                                    <div class="edit_button_row">
                                        {#if editTitle}
                                            <div class="edit_button_row_input">
                                                <Textfield
                                                    bind:value={selected["title"]}
                                                    label="Your report title"
                                                    input$rows={4}
                                                    textarea
                                                    variant="outlined"
                                                    style="width: 100%;"
                                                    helperLine$style="width: 100%;"
                                                />
                                            </div>
                                            <div>
                                                <IconButton class="material-icons grey_button" size="button" on:click={() => (editTitle = false)}>
                                                    check
                                                </IconButton>
                                            </div>
                                        {:else}
                                            {#if selected["title"] != ""}
                                                <div class="head_5">
                                                    {selected["title"]}
                                                </div>
                                            {:else}
                                                <div class="grey_text">Enter a report title</div>
                                            {/if}

                                            <div>
                                                <IconButton class="material-icons grey_button" size="button" on:click={() => (editTitle = true)}>
                                                    create
                                                </IconButton>
                                            </div>
                                        {/if}
                                    </div>
                                </div>

                                <!-- Error type -->
                                <div class="spacing_vert_40">
                                    <div class="head_6">
                                        <b>Error Type</b> 
                                    </div>
                                    <div class="edit_button_row">
                                        {#if editErrorType}
                                            <div>
                                                {#each error_type_options as e}
                                                    <div style="display: flex; align-items: center;">
                                                        <FormField>
                                                            <Radio bind:group={selected["error_type"]} value={e.opt} on:change={updateErrorType} color="secondary" />
                                                            <span slot="label">
                                                                <b>{e.opt}</b> {e.descr}
                                                            </span>
                                                        </FormField>
                                                    </div>
                                                {/each}
                                            </div>
                                        {:else}
                                            {#if selected["error_type"] != ""}
                                                <div>
                                                    <p>{selected["error_type"]}</p>
                                                </div>
                                            {:else}
                                                <div class="grey_text">Select an error type</div>
                                            {/if}
                                            
                                            <div>
                                                <IconButton class="material-icons grey_button" size="button" on:click={() => (editErrorType = true)}>
                                                    create
                                                </IconButton>
                                            </div>
                                        {/if}
                                    </div>
                                </div>
                                
                                <!-- Evidence -->
                                <div class="spacing_vert_40">
                                    <div class="head_6">
                                        <b>Evidence</b>
                                    </div>
                                    {#key cur_open_evidence}
                                    <div>
                                        {#if cur_open_evidence.length > 0}
                                        <ClusterResults 
                                            cluster={cur_topic} 
                                            model={model} 
                                            data={{"cluster_comments": cur_open_evidence}} 
                                            show_vis={false} 
                                            show_checkboxes={false} 
                                            table_width_pct={100} 
                                            rowsPerPage={25} 
                                            table_id={"panel"}
                                        />
                                        {:else}
                                            <p class="grey_text">
                                                Add examples from the main panel to see them here!
                                            </p>
                                        {/if}
                                    </div>
                                    {/key}
                                </div>

                                <div class="spacing_vert_40">
                                    <div class="head_6">
                                        <b>Summary/Suggestions</b>
                                    </div>
                                    <div class="spacing_vert">
                                        <Textfield
                                            style="width: 100%;"
                                            helperLine$style="width: 100%;"
                                            input$rows={8}
                                            textarea
                                            bind:value={selected["text_entry"]}
                                            label="My current hunch is that..."
                                        >
                                        </Textfield>
                                    </div>
                                    
                                </div>

                                <div class="spacing_vert_40 spacing_vert_100_bottom">
                                    <div class="head_6">
                                        <b>Audit Category</b>
                                    </div>
                                    <div>
                                        <Select bind:value={selected["sep_selection"]} label="Audit category" style="width: 90%">
                                            {#each all_sep_options as opt}
                                                <Option value={opt}>{opt}</Option>
                                            {/each}
                                        </Select>
                                    </div>
                                </div>
                            </div>
                            {/if}
                        </main>
                    </AppContent>
                </div>
                {/if}
            {:catch error}
                <p style="color: red">{error.message}</p>
            {/await}
        </div>

        <div class="panel_footer">
            <div class="panel_footer_contents">
                <!-- New button -->
                <Button 
                    on:click={handleNewReport} 
                    variant="outlined" 
                >
                    <Label>New</Label>
                </Button>

                <!-- Delete button -->
                <!-- <Button 
                    on:click={handleDeleteReport} 
                    variant="outlined" 
                >
                    <Label>Delete</Label>
                </Button> -->

                <!-- Save button -->
                <Button 
                    on:click={handleSaveReport} 
                    variant="outlined" 
                >
                    <Label>
                        {#await promise_save}
                            <CircularProgress style="height: 13.5px; width: 13.5px;" indeterminate />
                        {:then result}
                            {#if result && save_check_visible}
                            <span transition:fade>
                                <Icon class="material-icons">check</Icon>
                            </span>
                            {/if}
                        {:catch error}
                            <span style="color: red">{error.message}</span>
                        {/await}
                        Save
                    </Label>
                </Button>

                <!-- Send to Avid button -->
                {#key has_complete_report}
                <Button 
                    on:click={handleSubmitReport} 
                    variant="outlined"
                >
                    <Label>Send Reports to AVID</Label>
                </Button>
                {/key}
            </div>

            <div class="feedback_section">
                <a href="https://forms.gle/vDXchpbBFjDeKjJA6" target="_blank" class="feedback_link">
                    Share feedback about the tool
                </a>
            </div>
        </div>
        {/if}
    </div>
</div>

<style>
    .panel_contents {
        padding: 0 20px;
        overflow-y: auto;
        top: 150px;
        position: relative;
        height: 82%;
    }
    .panel_contents2 {
        padding-left: 10px;
    }

    .panel_header {
        position: fixed;
        width: 30%;
        border-bottom: 1px solid #d7d7d7; /* c5c5c5 */
        background: #f3f3f3;
        z-index: 11;
    }

    .panel_footer {
        position: fixed;
        width: 30%;
        border-top: 1px solid #d7d7d7;
        background: #f3f3f3;
        z-index: 11;
        bottom: 0;
        padding: 5px 0px;
    }
    .panel_footer_contents {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 5px 0px 10px 0px;
    }
    .feedback_section {
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    .feedback_link {
        color: var(--mdc-theme-secondary);
        font-size: 10px;
        text-decoration: underline;
    }

    :global(.mdc-button.user_button) {
        float: right;
        margin-right: 20px;
        max-width: 200px;
    }

    :global(.mdc-button.user_button span) {
        text-overflow: ellipsis;
        white-space: nowrap;
        overflow: hidden;
    }

    .page_header {
        width: 100%;
        background: #e3d6fd;
        /* padding: 21px 0; */
        /* border-bottom: 1px solid #e3d6fd; */
        padding: 10.5px 0;
        position: relative;
        display: inline-block;
    }

    .page_header:before {
        content: '';
        border-right: 1px solid rgb(0 0 0 / 7%);
        position: absolute;
        height: 80%;
        top: 10%;
        right: 0;
    }

    .hypotheses_header {
        display: inline-block;
        width: 100%;
        padding: 10px 0;
        vertical-align: middle;
    }
</style>

<script lang="ts">
    import { VegaLite } from "svelte-vega";
    import type { View } from "svelte-vega";
    import DataTable, {
        Head,
        Body,
        Row,
        Cell,
        Pagination,
    } from "@smui/data-table";
    import Select, { Option } from "@smui/select";
    import IconButton from "@smui/icon-button";
    import Button from "@smui/button";
    import { Label } from "@smui/common";
    import Checkbox from '@smui/checkbox';
    import Radio from '@smui/radio';
    import FormField from '@smui/form-field';
    import Tooltip, { Wrapper } from '@smui/tooltip';
    import LayoutGrid, { Cell as LGCell } from "@smui/layout-grid";
    import Card, { Content } from '@smui/card';

    import HelpTooltip from "./HelpTooltip.svelte";
    import { topic_chosen } from './stores/cur_topic_store.js';
    import { new_evidence } from './stores/new_evidence_store.js';
    import { open_evidence } from './stores/open_evidence_store.js';

    export let data;
    export let cluster;
    export let clusters = null;
    export let model;
    export let show_vis = true;
    export let show_checkboxes = true;
    export let table_width_pct = 80;
    export let rowsPerPage = 10;
    export let evidence;
    export let table_id;
    export let use_model = true;
    export let show_agree_disagree = false;

    let N_COMMENTS = 500;
    let show_num_ratings = false;
    let show_your_decision_ratings = false;
    let show_step2_info = false;

    let comment_table_style;
    if (show_checkboxes) {
        comment_table_style = ""
    } else {
        comment_table_style = "comment_table_small"
    }

    // Handle Altair selections
    let selected_comment_id = 0;
    window.addEventListener("popstate", function (event) {
    //your code goes here on location change 
        let cur_url = window.location.href;
        let cur_url_elems = cur_url.split("#");
        if (cur_url_elems.length > 0) {
            let path = cur_url_elems[2];
            if (path == "comment") {
                let comment_id = cur_url_elems[1].split("/")[0];
                selected_comment_id = parseInt(comment_id);
                let table_ind = null;
                for (let i = 0; i < items.length; i++) {
                    if (items[i]["id"] == selected_comment_id) {
                        table_ind = i;
                        break;
                    }
                }
                currentPage = table_ind / rowsPerPage;
            } else if (path == "topic") {
                let topic = cur_url_elems[1].split("/")[0];
                topic_chosen.update((value) => topic); // update in store
            }
            // window.history.replaceState({}, document.title, "/"); // remove URL parameter
        }
    });

    // Cluster Overview Plot
    let cluster_overview_data = null;
    let cluster_overview_spec = null;
    let cluster_overview_view = null;
    if (show_vis) {
        let cluster_overview_json = data["cluster_overview_plot_json"];
        cluster_overview_data =
            cluster_overview_json["datasets"][
                cluster_overview_json["layer"][0]["data"]["name"]
            ];
        cluster_overview_spec = cluster_overview_json;
        cluster_overview_view = null;
    }
    
    type ClusterComment = {
        id: number;
        comment: string;
        user_decision: string;
        user_rating: number;
        system_decision: string;
        system_rating: number;
        user_color: string;
        system_color: string;
        error_type: string;
        error_color: string;
        judgment: string;
        toxicity_category: string;
    };
    let items: ClusterComment[];

    let selected = [];

    // Pagination
    let currentPage = 0;
    $: start = currentPage * rowsPerPage;
    $: end = Math.min(start + rowsPerPage, items.length);
    $: slice = items.slice(start, end);
    $: lastPage = Math.max(Math.ceil(items.length / rowsPerPage) - 1, 0);

    $: if (currentPage > lastPage) {
        currentPage = lastPage;
    }

    let set_length = 0;
    // if (typeof(data["cluster_comments"] == "string")) {
    if (show_checkboxes) {
        items = JSON.parse(data["cluster_comments"]);
        set_length = data["topic_df_ids"].length;
    } else {
        items = data["cluster_comments"];
        set_length = items.length;
    }
    
    let cur_open_evidence;
    open_evidence.subscribe(value => {
		cur_open_evidence = value;
	});

    function saveToEvidence() {
        new_evidence.update((value) => []); // clear prior evidence
        selected.forEach(function(s) {
            if (!cur_open_evidence.includes(s)) {
                new_evidence.update((value) => s); // update in store
            }
        });
        selected = [];

        // Clear highlighted rows
        let rows = document.getElementsByTagName("tr");
        let row_list = Array.prototype.slice.call(rows);
        row_list.forEach(function(r) {
            r.classList.remove("mdc-data-table__row--selected");
        });

        let checkbox_header_divs = document.getElementsByClassName("mdc-data-table__header-row-checkbox");
        let checkbox_header_list = Array.prototype.slice.call(checkbox_header_divs);
        checkbox_header_list.forEach(function(c) {
            let c_input = c.getElementsByTagName("input");
            for (let i = 0; i < c_input.length; i++) {
                c_input[i].setAttribute("data-indeterminate", "false");
                c_input[i].indeterminate = false;
            }
        });
    }

    function handleAdd(comment_to_remove) {
        new_evidence.update((value) => []); // clear prior evidence
        if (!cur_open_evidence.includes(comment_to_remove)) {
            new_evidence.update((value) => comment_to_remove); // update in store
        }
    }

    function handleRemove(comment_to_remove) {
        // Update local open evidence
        cur_open_evidence = cur_open_evidence.filter(item => item.comment != comment_to_remove)
        // Update open evidence in store
        open_evidence.update((value) => cur_open_evidence);
    }
</script>

<div class="padding-top: 30px;">
    {#if show_vis}
        <div>
            <span class="head_6">Topic overview: {cluster}</span>
            <IconButton 
                class="material-icons grey_button"
                size="normal"
                on:click={() => (show_step2_info = !show_step2_info)}
            >
                help_outline
            </IconButton>
        </div>
        {#if N_COMMENTS < set_length}
            <p>Showing a random sample of {N_COMMENTS} comments (out of {set_length} comments)</p>
        {:else}
            <p>Showing all {set_length} comments</p>
        {/if}

        {#if show_step2_info}
            <LayoutGrid>
                <LGCell span={8}>
                    <div class="card-container">
                        <Card variant="outlined" padded>
                            <p class="mdc-typography--button"><b>Interpreting this visualization</b></p>
                            <ul>
                                <li>
                                    This plot has the same layout as the <b>All Topics</b> visualization, but now, each <b>box</b> in this plot represents an <b>individual comment</b> that belongs to your <b>selected topic area</b>.
                                </li>
                                <li>
                                    The <b>x-axis</b> represents our prediction of <b>your</b> toxicity rating for each comment (we'll call these "your ratings")
                                    <ul>
                                        <li>
                                            The <b>left side</b> (white background) is the <b>Non-toxic</b> side (comments that'll be allowed to remain)
                                        </li>
                                        <li>
                                            The <b>right side</b> (grey background) is the <b>Toxic</b> side (comments that will be deleted)
                                        </li>
                                        <li>
                                            Comment boxes are plotted along the x-axis based on our prediction of your toxicity rating for that comment
                                        </li>
                                    </ul>
                                </li>
                    
                                <li>
                                    The <b>color</b> of the box indicates the <b>system's rating</b> for the same comment; you may want to focus on the <b>red-colored boxes</b> that indicate <b>disagreements</b> between "your ratings" and the system's ratings
                                </li>
                            </ul>
                        </Card>
                    </div>
                </LGCell>
            </LayoutGrid>
        {/if}

        <div class="row">
            <div class="col s8">
                <VegaLite
                    {cluster_overview_data}
                    spec={cluster_overview_spec}
                    bind:view={cluster_overview_view}
                />
            </div>
        </div>
    {/if}
    
    {#if show_checkboxes}
    <h6>Comments</h6>
    {/if}
    <!-- Display options -->
    {#if show_checkboxes}
    <div>
        Numerical ratings:
        <FormField>
            <Radio bind:group={show_num_ratings} value={true} color="secondary" />
            <span slot="label">Show</span>
        </FormField>
        <FormField>
            <Radio bind:group={show_num_ratings} value={false} color="secondary" />
            <span slot="label">Hide</span>
        </FormField>
    </div>
    {#if use_model}
    <div>
        Our prediction of your decision + ratings:
        <FormField>
            <Radio bind:group={show_your_decision_ratings} value={true} color="secondary" />
            <span slot="label">Show</span>
        </FormField>
        <FormField>
            <Radio bind:group={show_your_decision_ratings} value={false} color="secondary" />
            <span slot="label">Hide</span>
        </FormField>
    </div>
    {/if}

    <!-- <Wrapper>
        <IconButton class="material-icons" size="button" disabled>help_outline</IconButton>
        <Tooltip>White = Non-toxic, Grey = Toxic</Tooltip>
    </Wrapper> -->
    {/if}

    {#key evidence}
    <div class="comment_table {comment_table_style}">
        <DataTable
            table$aria-label="Comments in the topic cluster"
            style="width: {table_width_pct}%;"
        >
            <Head>
                <Row>
                    <!-- {#if show_checkboxes}
                    <Cell checkbox>
                        <Checkbox />
                    </Cell>
                    {/if} -->

                    <Cell style="width: 50%">Comment</Cell>

                    {#if show_your_decision_ratings}
                    <Cell>Our prediction<br>of your decision</Cell>
                    {#if show_num_ratings}
                    <Cell>Our prediction<br>of your rating</Cell>
                    {/if}
                    {/if}
                    
                    <Cell>
                        System<br>decision<br>
                        {#if show_checkboxes}
                        <span style="font-size:12px; max-width:125px">White = Non-toxic, <br>Grey = Toxic</span>
                        {/if}
                    </Cell>
                    {#if show_num_ratings}
                    <Cell>System<br>rating</Cell>
                    {/if}

                    {#if show_checkboxes}
                    {#if use_model}
                    <Cell>
                        Potential error<br>type<br>
                        {#if show_checkboxes}
                        <span style="font-size:12px; max-width:125px">Darker red = Greater <br>potential system error</span>
                        {/if}
                    </Cell>

                    <Cell>Potential toxicity<br>categories</Cell>
                    {/if}
                    {/if}
                    
                    {#if show_agree_disagree}
                    <Cell>Do you agree<br>with the system?</Cell>
                    {/if}

                    {#if !show_checkboxes}
                    <Cell>Remove</Cell>
                    {/if}

                    {#if show_checkboxes}
                    <Cell>Add<br>Evidence</Cell>
                    {/if}
                </Row>
            </Head>
            <Body>
                {#each slice as item (item.id + table_id)}
                    <Row>
                        <!-- {#if show_checkboxes}
                        <Cell checkbox>
                            <Checkbox
                                bind:group={selected}
                                value={{
                                    "comment": item.comment, 
                                    "user_color": item.user_color, 
                                    "user_decision": item.user_decision, 
                                    "user_rating": item.user_rating, 
                                    "system_color": item.system_color, 
                                    "system_decision": item.system_decision,
                                    "system_rating": item.system_rating, 
                                    "error_type": item.error_type, 
                                    "error_color": item.error_color,
                                    "toxicity_category": item.toxicity_category,
                                    "judgment": item.judgment, 
                                    "id": item.id
                                }}
                                valueKey={item.comment}
                            />
                        </Cell>
                        {/if} -->

                        <Cell>
                            {item.comment}
                        </Cell>

                        {#if show_your_decision_ratings}
                        <Cell style="background-color: {item.user_color}; border-left: 1px solid rgba(0,0,0,.12); border-right: 1px solid rgba(0,0,0,.12); border-collapse: collapse;">
                            {item.user_decision}
                        </Cell>
                        {#if show_num_ratings}
                        <Cell style="background-color: {item.user_color}; border-left: 1px solid rgba(0,0,0,.12); border-right: 1px solid rgba(0,0,0,.12); border-collapse: collapse;">
                            {item.user_rating}
                        </Cell>
                        {/if}
                        {/if}

                        <Cell style="background-color: {item.system_color}; border-left: 1px solid rgba(0,0,0,.12); border-right: 1px solid rgba(0,0,0,.12); border-collapse: collapse;">
                            {item.system_decision}
                        </Cell>
                        {#if show_num_ratings}
                        <Cell style="background-color: {item.system_color}; border-left: 1px solid rgba(0,0,0,.12); border-right: 1px solid rgba(0,0,0,.12); border-collapse: collapse;">
                            {item.system_rating}
                        </Cell>
                        {/if}

                        {#if show_checkboxes}
                        {#if use_model}
                        <Cell style="background-color: {item.error_color}; border-left: 1px solid rgba(0,0,0,.12); border-right: 1px solid rgba(0,0,0,.12); border-collapse: collapse;">
                            {item.error_type}
                        </Cell>

                        <Cell>
                            {item.toxicity_category}
                        </Cell>
                        {/if}
                        {/if}
                        
                        {#if show_agree_disagree}
                        <Cell>
                            <div>
                                <FormField>
                                    <Radio
                                      bind:group={item.judgment}
                                      value={"Agree"}
                                    />
                                    <span slot="label">Agree</span>
                                </FormField>
                            </div>
                            <div>
                                <FormField>
                                    <Radio
                                      bind:group={item.judgment}
                                      value={"Disagree"}
                                    />
                                    <span slot="label">Disagree</span>
                                </FormField>
                            </div>
                        </Cell>
                        {/if}

                        {#if !show_checkboxes}
                        <Cell>
                            <IconButton class="material-icons grey_button" on:click={() => handleRemove(item.comment)}>
                                remove_circle_outline
                            </IconButton>
                        </Cell>
                        {/if}

                        {#if show_checkboxes}
                        <Cell>
                            <IconButton class="material-icons grey_button" on:click={() => handleAdd(item)}>
                                add_circle_outline
                            </IconButton>
                        </Cell>
                        {/if}
                    </Row>
                {/each}
            </Body>

            <!-- Table pagination -->
            <Pagination slot="paginate">
                <svelte:fragment slot="rowsPerPage">
                    <Label>Rows Per Page</Label>
                    <Select variant="outlined" bind:value={rowsPerPage} noLabel>
                        <Option value={5}>5</Option>
                        <Option value={10}>10</Option>
                        <Option value={25}>25</Option>
                        <Option value={100}>100</Option>
                    </Select>
                </svelte:fragment>
                <svelte:fragment slot="total">
                    {start + 1}-{end} of {items.length}
                </svelte:fragment>

                <IconButton
                    class="material-icons"
                    action="first-page"
                    title="First page"
                    on:click={() => (currentPage = 0)}
                    disabled={currentPage === 0}>first_page</IconButton
                >
                <IconButton
                    class="material-icons"
                    action="prev-page"
                    title="Prev page"
                    on:click={() => currentPage--}
                    disabled={currentPage === 0}>chevron_left</IconButton
                >
                <IconButton
                    class="material-icons"
                    action="next-page"
                    title="Next page"
                    on:click={() => currentPage++}
                    disabled={currentPage === lastPage}
                    >chevron_right</IconButton
                >
                <IconButton
                    class="material-icons"
                    action="last-page"
                    title="Last page"
                    on:click={() => (currentPage = lastPage)}
                    disabled={currentPage === lastPage}>last_page</IconButton
                >
            </Pagination>
        </DataTable>
    </div>
    {/key}

    <!-- {#if show_checkboxes}
    <div class="spacing_vert">
        <Button on:click={saveToEvidence} disabled={selected.length == 0} variant="outlined">
            <Label>Save {selected.length} to evidence</Label>
        </Button>
    </div>
    {/if} -->
    
    <!-- Old visualization -->
    <!-- {#if show_vis}
        <div style="margin-top: 500px">
            <table>
                <tbody>
                    <tr class="custom-blue">
                        <td class="bold">
                            Compared to the system, YOUR labels are on average...
                        </td>
                        <td>
                            <span class="bold-large"
                                >{data["user_perf_rounded"]} points
                                {data["user_direction"]}</span
                            >
                            for this cluster
                        </td>
                    </tr>
                    <tr>
                        <td class="bold">
                            Compared to the system, OTHER USERS' labels are on
                            average...
                        </td>
                        <td>
                            <span class="bold-large"
                                >{data["other_perf_rounded"]} points
                                {data["other_direction"]}</span
                            >
                            for this cluster (based on {data["n_other_users"]} randomly-sampled
                            users)
                        </td>
                    </tr>
                    <tr>
                        <td class="bold"> Odds ratio </td>
                        <td>
                            <span class="bold-large">{data["odds_ratio"]}</span><br />
                            {data["odds_ratio_explanation"]}
                        </td>
                    </tr>
                </tbody>
            </table>
        
            <h6>Cluster examples</h6>
            <div class="row">
                <div class="col s12">
                    <div id="cluster_results_elem">
                        {@html data["cluster_examples"]}
                    </div>
                </div>
            </div>
        </div>
    {/if} -->
</div>

<style>
    /* Styles for table */
    :global(html) {
        height: auto;
        width: auto;
        position: static;
    }
    :global(#sapper),
    :global(body) {
        display: block;
        height: auto;
    }
</style>

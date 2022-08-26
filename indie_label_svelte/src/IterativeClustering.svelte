<script>
    import Section from "./Section.svelte";
    import ClusterResults from "./ClusterResults.svelte";
    import Button, { Label } from "@smui/button";
    import Textfield from "@smui/textfield";
    import LayoutGrid, { Cell } from "@smui/layout-grid";
    import LinearProgress from "@smui/linear-progress";
    import Chip, { Set, Text } from '@smui/chips';

    export let clusters;
    export let personalized_model;
    export let evidence;
    export let width_pct = 80;

    let topic_df_ids = [];
    let promise_iter_cluster = Promise.resolve(null);
    let keyword = null;
    let n_neighbors = null;
    let cur_iter_cluster = null;
    let history = [];

    async function getIterCluster(search_type) {
        let req_params = {
            cluster: cur_iter_cluster,
            topic_df_ids: topic_df_ids,
            n_examples: 500, // TEMP
            pers_model: personalized_model,
            example_sort: "descending", // TEMP
            comparison_group: "status_quo", // TEMP
            search_type: search_type,
            keyword: keyword,
            n_neighbors: n_neighbors,
        };
        console.log("topic_df_ids", topic_df_ids);
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_cluster_results?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        // if (data["cluster_comments"] == null) {
        //     return false
        // }
        topic_df_ids = data["topic_df_ids"];
        return data;
    }

    function findCluster() {
        promise_iter_cluster = getIterCluster("cluster");
        history = history.concat("bulk-add cluster: " + cur_iter_cluster);
    }

    function findNeighbors() {
        promise_iter_cluster = getIterCluster("neighbors");
        history = history.concat("find " + n_neighbors + " neighbors");
    }

    function findKeywords() {
        promise_iter_cluster = getIterCluster("keyword");
        history = history.concat("keyword search: " + keyword);
    }
</script>

<div>
    <div>
        <!-- <h6>Hunch {ind} examples</h6> -->
        <div>
            <h6>Search Settings</h6>
            <!-- Start with cluster -->
            <!-- <div class="">
                <Section
                    section_id="iter_cluster"
                    section_title="Bulk-add cluster"
                    section_opts={clusters}
                    bind:value={cur_iter_cluster}
                    width_pct={100}
                />
                <Button
                    on:click={findCluster}
                    variant="outlined"
                    class="button_float_right"
                    disabled={cur_iter_cluster == null}
                >
                    <Label>Search</Label>
                </Button>
            </div> -->

            <!-- Manual keyword -->
            <div class="spacing_vert">
                <Textfield
                    bind:value={keyword}
                    label="Keyword search"
                    variant="outlined"
                    style="width: {width_pct}%"
                />
                <Button
                    on:click={findKeywords}
                    variant="outlined"
                    class="button_float_right spacing_vert"
                    disabled={keyword == null}
                >
                    <Label>Search</Label>
                </Button>
            </div>

            <!-- Find neighbors of current set -->
            <div class="spacing_vert">
                <Textfield
                    bind:value={n_neighbors}
                    label="Number of neighbors to retrieve"
                    type="number"
                    min="1"
                    max="50"
                    variant="outlined"
                    style="width: {width_pct}%"
                />
                <Button
                    on:click={findNeighbors}
                    variant="outlined"
                    class="button_float_right spacing_vert"
                    disabled={n_neighbors == null}
                >
                    <Label>Search</Label>
                </Button>
            </div>
        </div>
    </div>

    {#await promise_iter_cluster}
        <div class="app_loading" style="width: {width_pct}%">
            <LinearProgress indeterminate />
        </div>
    {:then iter_cluster_results}
        {#if iter_cluster_results}
            {#if history.length > 0}
                <div class="bold" style="padding-top:40px;">Search History</div>
                <Set chips={history} let:chip choice>
                    <Chip {chip}>
                        <Text>{chip}</Text>
                    </Chip>
                </Set>
            {/if}
            {#if iter_cluster_results.cluster_comments != null}
                <ClusterResults
                    cluster={""}
                    clusters={clusters}
                    model={personalized_model}
                    data={iter_cluster_results}
                    show_vis={false}
                    table_width_pct={80}
                    bind:evidence={evidence}
                    on:change
                />
            {:else}
                <div class="bold" style="padding-top:40px;">
                    No results found
                </div>
            {/if}
        {/if}
    {:catch error}
        <p style="color: red">{error.message}</p>
    {/await}
</div>

<style>
</style>

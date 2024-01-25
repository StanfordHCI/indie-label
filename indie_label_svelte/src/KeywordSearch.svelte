<script>
    import ClusterResults from "./ClusterResults.svelte";
    import { error_type } from './stores/error_type_store.js';
    
    import Button, { Label } from "@smui/button";
    import Textfield from "@smui/textfield";
    import LinearProgress from "@smui/linear-progress";    

    export let clusters;
    export let personalized_model;
    export let cur_user;
    export let evidence;
    export let width_pct = 80;
    export let use_model = true;

    let topic_df_ids = [];
    let promise_iter_cluster = Promise.resolve(null);
    let keyword = null;
    let cur_iter_cluster = null;
    let history = [];

    let cur_error_type;
    error_type.subscribe(value => {
		cur_error_type = value;
	});

    async function getIterCluster(search_type) {
        let req_params = {
            cluster: cur_iter_cluster,
            topic_df_ids: topic_df_ids,
            cur_user: cur_user,
            pers_model: personalized_model,
            example_sort: "descending", // TEMP
            comparison_group: "status_quo", // TEMP
            search_type: search_type,
            keyword: keyword,
            error_type: cur_error_type,
        };
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_cluster_results?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        topic_df_ids = data["topic_df_ids"];
        return data;
    }

    function findKeywords() {
        history = [];
        topic_df_ids = [];
        promise_iter_cluster = getIterCluster("keyword");
        history = history.concat("keyword search: " + keyword);
    }

    function clearHistory() {
        history = [];
        promise_iter_cluster = Promise.resolve(null);
        keyword = "";
        topic_df_ids = [];
    }
</script>

<div>
    <div>
        <div>
            <h6>Keyword Search</h6>
            <!-- Manual keyword -->
            <div class="spacing_vert edit_button_row" style="width: 90%; justify-content: space-between">
                <Textfield
                    bind:value={keyword}
                    label="Your keyword or phrase"
                    variant="outlined"
                    style="width: 60%"
                />
                <Button
                    on:click={findKeywords}
                    variant="outlined"
                    class="spacing_vert"
                    disabled={keyword == null}
                >
                    <Label>Search</Label>
                </Button>                
                <!-- <Button
                    on:click={clearHistory}
                    variant="outlined"
                    class="spacing_vert"
                    disabled={history.length == 0}
                >
                    <Label>Clear Search</Label>
                </Button> -->
            </div>

            <!-- {#if history.length > 0}
                <div class="head_6">Search History</div>
                <Set chips={history} let:chip choice>
                    <Chip {chip}>
                        <Text>{chip}</Text>
                    </Chip>
                </Set>
            {/if} -->
        </div>
    </div>

    {#await promise_iter_cluster}
        <div class="app_loading" style="width: {width_pct}%">
            <LinearProgress indeterminate />
        </div>
    {:then iter_cluster_results}
        {#if iter_cluster_results}
            {#if iter_cluster_results.cluster_comments != null}
                <ClusterResults
                    cluster={""}
                    clusters={clusters}
                    model={personalized_model}
                    data={iter_cluster_results}
                    show_vis={false}
                    table_width_pct={90}
                    table_id={"keyword"}
                    use_model={use_model}
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

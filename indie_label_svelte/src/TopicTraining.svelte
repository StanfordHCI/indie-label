<script lang="ts">
    import { onMount } from "svelte";
    import ModelPerf from "./ModelPerf.svelte";
    import Button, { Label } from "@smui/button";
    import DataTable, { Head, Body, Row, Cell } from "@smui/data-table";
    import LinearProgress from '@smui/linear-progress';
    import { model_chosen } from './stores/cur_model_store.js';

    export let topic;
    export let model_name = null;
    export let cur_user;

    let to_label = {};
    let promise = Promise.resolve(null);

    // Get current model
    if (model_name == null) {
        model_chosen.subscribe(value => {
            model_name = value;
        });
    }

    function getCommentsToLabel() {
        let req_params = {
            topic: topic,
        };
        let params = new URLSearchParams(req_params).toString();
        fetch("./get_comments_to_label_topic?" + params)
            .then((r) => r.text())
            .then(function (r_orig) {
                let r = JSON.parse(r_orig);
                // Append comment rows to to_label object
                r["to_label"].forEach((key) => (to_label[key] = null));
            });
    }

    onMount(async () => {
        getCommentsToLabel();
    });

    function handleLoadCommentsButton() {
        getCommentsToLabel();
    }

    function handleTrainModelButton() {
        promise = trainModel();
    }

    function getRatings() {
        // Get rating for each comment from HTML elems
        let ratings = {};
        Object.entries(to_label).forEach(function ([comment, orig_rating], i) {
            var radio_btns = document.getElementsByName(
                "comment_" + i.toString()
            );
            let length = radio_btns.length;
            for (var i = 0; i < length; i++) {
                if (radio_btns[i].checked) {
                    ratings[comment] = radio_btns[i].value;
                    break;
                }
            }
        });
        return ratings;
    }

    async function trainModel() {
        let ratings = getRatings();
        ratings = JSON.stringify(ratings);

        let req_params = {
            model_name: model_name,
            ratings: ratings,
            user: cur_user,
            topic: topic,
        };

        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_personalized_model_topic?" + params); // TODO
        const text = await response.text();
        const data = JSON.parse(text);
        // to_label = data["ratings_prev"];
        model_name = data["new_model_name"];
        model_chosen.update((value) => model_name);

        return data;
    }
</script>

<div>
{#if topic}
    <div class="label_table_expandable spacing_vert">
        <DataTable
            table$aria-label="Comments to label"
            style="width: 100%;"
            stickyHeader
        >
            <Head>
                <Row>
                    <Cell style="width: 50%">Comment</Cell>
                    <Cell style="background-color: #c3ecdb">
                        0: <br>Not-at-all toxic<br>(Keep)<br>
                    </Cell>
                    <Cell style="background-color: white">
                        1: <br>Slightly toxic<br>(Keep)<br>
                    </Cell>
                    <Cell style="background-color: #ffa894">
                        2: <br>Moderately toxic<br>(Delete)<br>
                    </Cell>
                    <Cell style="background-color: #ff7a5c">
                        3: <br>Very toxic<br>(Delete)<br>
                    </Cell>
                    <Cell style="background-color: #d62728">
                        4: <br>Extremely toxic<br>(Delete)<br>
                    </Cell>
                    <Cell style="background-color: #808080">
                        <br>Unsure<br>(Skip)<br>
                    </Cell>
                </Row>
            </Head>
            <Body>
                {#if to_label}
                    {#each Object.keys(to_label) as comment, i}
                        <Row>
                            <Cell>
                                <div class="spacing_vert">{comment}</div>
                            </Cell>
                            <Cell>
                                <label>
                                    <input
                                        name="comment_{i}"
                                        type="radio"
                                        value="0"
                                        checked={to_label[comment] == "0"}
                                    />
                                    <span />
                                </label>
                            </Cell>
                            <Cell>
                                <label>
                                    <input
                                        name="comment_{i}"
                                        type="radio"
                                        value="1"
                                        checked={to_label[comment] == "1"}
                                    />
                                    <span />
                                </label>
                            </Cell>
                            <Cell>
                                <label>
                                    <input
                                        name="comment_{i}"
                                        type="radio"
                                        value="2"
                                        checked={to_label[comment] == "2"}
                                    />
                                    <span />
                                </label>
                            </Cell>
                            <Cell>
                                <label>
                                    <input
                                        name="comment_{i}"
                                        type="radio"
                                        value="3"
                                        checked={to_label[comment] == "3"}
                                    />
                                    <span />
                                </label>
                            </Cell>
                            <Cell>
                                <label>
                                    <input
                                        name="comment_{i}"
                                        type="radio"
                                        value="4"
                                        checked={to_label[comment] == "4"}
                                    />
                                    <span />
                                </label>
                            </Cell>
                            <Cell>
                                <label>
                                    <input
                                        name="comment_{i}"
                                        type="radio"
                                        value="-1"
                                        checked={to_label[comment] == "-1"}
                                    />
                                    <span />
                                </label>
                            </Cell>
                        </Row>
                    {/each}
                {/if}
            </Body>
        </DataTable>
    </div>

    <div class="">
        <Button on:click={handleTrainModelButton} variant="outlined">
            <Label>Tune Model</Label>
        </Button>
        <Button on:click={handleLoadCommentsButton} variant="outlined">
            <Label>Fetch More Comments To Label</Label>
        </Button>
    </div>

    <!-- Performance -->
    {#await promise}
        <div class="app_loading spacing_vert_20">
            <LinearProgress indeterminate />
        </div>
    {:then perf_results}
        {#if perf_results}
            <div class="spacing_vert_20">
                Model for the topic {topic} has been successfully tuned. You can now proceed to explore this topic.
            </div>
        {/if}
    {:catch error}
        <p style="color: red">{error.message}</p>
    {/await}
{/if}
</div>

<style>
</style>

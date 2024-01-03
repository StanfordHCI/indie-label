<script lang="ts">
    import { onMount } from "svelte";
    import ModelPerf from "./ModelPerf.svelte";
    import Button, { Label } from "@smui/button";
    import DataTable, { Head, Body, Row, Cell } from "@smui/data-table";
    import LinearProgress from '@smui/linear-progress';

    import { model_chosen } from './stores/cur_model_store.js';

    export let mode;
    export let model_name;
    export let cur_user;

    let to_label = {};
    let promise = Promise.resolve(null);
    let n_complete_ratings;
    let n_unsure_ratings;
    let show_comments_labeled_count = false;

    function getCommentsToLabel(cur_mode, n) {
        if (cur_mode == "train") {
            let req_params = {
                n: n,
            };
            let params = new URLSearchParams(req_params).toString();
            fetch("./get_comments_to_label?" + params)
                .then((r) => r.text())
                .then(function (r_orig) {
                    let r = JSON.parse(r_orig);
                    r["to_label"].forEach((key) => (to_label[key] = null));
                });
        } else if (cur_mode == "view") {
            if (model_name != "" && model_name != undefined) {
                promise = getModel(cur_mode);
            }
        }
    }
    onMount(async () => {
        getCommentsToLabel(mode, 40);
    });

    function handleLoadCommentsButton(n = 5) {
        getCommentsToLabel("train", n);
    }

    function handleTrainModelButton() {
        getCompleteRatings();
        promise = getModel("train");
    }

    function getCompleteRatings() {
        let ratings = getRatings();
        let complete_ratings = Object.entries(ratings).filter(([key, value]) => value != "-1");
        let unsure_ratings = Object.entries(ratings).filter(([key, value]) => value == "-1");
        n_complete_ratings = complete_ratings.length;
        n_unsure_ratings = unsure_ratings.length;
    }

    function getRatings() {
        // Get rating for each comment
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

    async function getModel(cur_mode) {
        let ratings = null;
        if (cur_mode == "train") {
            ratings = getRatings();
            ratings = JSON.stringify(ratings);
        }

        let req_params = {
            model_name: model_name,
            ratings: ratings,
            mode: cur_mode,
            user: cur_user,
        };
        let params = new URLSearchParams(req_params).toString();
        const data = await fetch("./get_personalized_model?" + params)
            .then((r) => r.text())
            .then(function (text) {
                let data = JSON.parse(text);
                to_label = data["ratings_prev"];
                model_chosen.update((value) => model_name);
                return data;
            });
        return data;
    }
</script>

<div>
    <div class="label_table spacing_vert">
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
                                        on:click={() => handleLoadCommentsButton(1)}
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

    {#key n_complete_ratings}
    {#if n_complete_ratings}
    <div class="spacing_vert_40">
        <p>Number labeled: {n_complete_ratings}</p>
        <p>Number unsure: {n_unsure_ratings}</p>
    </div>
    {/if}
    {/key}

    <div class="spacing_vert_40">
        <Button on:click={handleTrainModelButton} variant="outlined">
            <Label>Train Model</Label>
        </Button>
        {#if show_comments_labeled_count}
        <Button on:click={getCompleteRatings} variant="outlined">
            <Label>Get Number of Comments Labeled</Label>
        </Button>
        {/if}
        <Button on:click={() => handleLoadCommentsButton(5)} variant="outlined">
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
                <ModelPerf data={perf_results} />
            </div>
        {/if}
    {:catch error}
        <p style="color: red">{error.message}</p>
    {/await}
</div>

<style>
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

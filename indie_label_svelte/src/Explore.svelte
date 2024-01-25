<script lang="ts">
    import { onMount } from "svelte";
    import Button, { Label } from "@smui/button";
    import LinearProgress from "@smui/linear-progress";
    import DataTable, {
        Head,
        Body,
        Row,
        Cell,
        Label,
        SortValue,
    } from "@smui/data-table";
    import IconButton from '@smui/icon-button';
    import Radio from "@smui/radio";
    import FormField from "@smui/form-field";

    let cur_examples = [];
    let promise = Promise.resolve(null);

    // let sort_examples = false;
    let sort = "rating";
    let sortDirection: Lowercase<keyof typeof SortValue> = "descending";

    function handleSort() {
        cur_examples.sort((a, b) => {
            const [aVal, bVal] = [a[sort], b[sort]][
                sortDirection === "ascending" ? "slice" : "reverse"
            ]();
            return Number(aVal) - Number(bVal);
        });
        cur_examples = cur_examples;
    }

    onMount(async () => {
        promise = getExamples();
    });

    function handleButton() {
        promise = getExamples();
    }
    
    async function getExamples() {
        let req_params = {
            n_examples: 20,
        };
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_explore_examples?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        cur_examples = JSON.parse(data["examples"]);
        return true;
    }
</script>

<svelte:head>
    <title>Explore</title>
</svelte:head>

<div class="panel">
    <div class="panel_contents">
        <div>
            <h3>Explore System</h3>
            <div style="width: 50%">
                <ul>
                    <li>
                        Take a few minutes to explore some examples of <b>comments on YouSocial</b> and the toxicity ratings provided by YouSocial's <b>content moderation system</b>.
                    </li>
                    <li>
                        You can optionally sort by the "System rating" by clicking on the arrow button in the header.
                    </li>
                    <li>
                        Feel free to click the button to fetch a new sample of examples if you'd like.
                    </li>
                </ul>
            </div>
            <!-- <div>
                Sort order:
                <FormField>
                    <Radio bind:group={sort_examples} value={false} color="secondary" />
                    <span slot="label">None</span>
                </FormField>
                <FormField>
                    <Radio bind:group={sort_examples} value={true} color="secondary" />
                    <span slot="label">System toxicity rating (descending)</span>
                </FormField>
            </div> -->
            <Button on:click={handleButton} variant="outlined" class="">
                <Label>Get another sample of examples</Label>
            </Button>
        </div>

        <div style="padding-top:50px">
            {#await promise}
                <div class="app_loading">
                    <LinearProgress indeterminate />
                </div>
            {:then examples}
                {#if cur_examples}
                    <DataTable
                        table$aria-label="Example list"
                        style="max-width: 100%;"
                        sortable
                        bind:sort
                        bind:sortDirection
                        on:SMUIDataTable:sorted={handleSort}
                    >
                        <Head>
                            <Row>
                                <Cell sortable={false}>
                                    <Label>Comment</Label>
                                </Cell>
                                <Cell sortable={false}>
                                    <Label>System decision</Label>
                                </Cell>
                                <Cell numeric columnId="rating" sortable={true}>
                                    <IconButton class="material-icons">arrow_upward</IconButton>
                                    <Label>System rating</Label>
                                </Cell>
                            </Row>
                        </Head>
                        <Body>
                            {#each cur_examples as ex (ex.item_id)}
                                <Row>
                                    <Cell>{ex.comment}</Cell>
                                    <Cell
                                        style="background-color: {ex.system_color}; border-left: 1px solid rgba(0,0,0,.12); border-right: 1px solid rgba(0,0,0,.12); border-collapse: collapse;"
                                    >
                                        {ex.system_decision}
                                    </Cell>
                                    <Cell numeric>{Number(ex.rating)}</Cell>
                                </Row>
                            {/each}
                        </Body>
                    </DataTable>
                {/if}
            {:catch error}
                <p style="color: red">{error.message}</p>
            {/await}
        </div>
    </div>
</div>

<style>
    .panel {
        width: 80%;
        padding: 50px;
    }
</style>

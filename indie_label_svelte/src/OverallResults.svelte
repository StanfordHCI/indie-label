<script lang="ts">
    import { VegaLite } from "svelte-vega";
	import type { View } from "svelte-vega";

    import IconButton from '@smui/icon-button';
    import LayoutGrid, { Cell } from "@smui/layout-grid";
    import Card, { Content } from '@smui/card';

    export let data;
    export let clusters;
    export let personalized_model;
    export let cluster = "";
    
    let show_step1_info = false;

    // Topic Overview Plot
    let topic_overview_json = data["overall_perf"]["topic_overview_plot_json"];
    let topic_overview_data = topic_overview_json["datasets"][topic_overview_json["layer"][0]["data"]["name"]];
    let topic_overview_spec = topic_overview_json;
    let topic_overview_view: View;

    // // Overall Histogram
    // let overall_hist_json = data["overall_perf"]["overall_hist_json"];
    // let overall_hist_data = overall_hist_json["datasets"][overall_hist_json["data"]["name"]];
    // let overall_hist_spec = overall_hist_json;
    // let overall_hist_view: View;

    // // Class-conditional Histogram
    // let class_cond_plot_json = data["overall_perf"]["class_cond_plot_json"];
    // let class_cond_plot_data = class_cond_plot_json["datasets"][class_cond_plot_json["data"]["name"]];
    // let class_cond_plot_spec = class_cond_plot_json;
    // let class_cond_plot_view: View;

</script>

<div>
    <div>
        <span class="head_6">All topics</span>
        <IconButton 
            class="material-icons grey_button"
            size="normal"
            on:click={() => (show_step1_info = !show_step1_info)}
        >
            help_outline
        </IconButton>
    </div>
    {#if show_step1_info}
        <LayoutGrid>
            <Cell span={8}>
                <div class="card-container">
                    <Card variant="outlined" padded>
                        <p class="mdc-typography--button"><b>Interpreting this visualization</b></p>
                        <ul>
                            <li>
                                Each <b>box</b> in this plot represents a set of comments that belong in a given <b>topic area</b>
                            </li>
                            <li>
                                The <b>x-axis</b> represents our prediction of <b>your</b> toxicity rating for each topic (we'll call these "your ratings")
                                <ul>
                                    <li>
                                        The <b>left side</b> (white background) is the <b>Non-toxic</b> side (comments that'll be allowed to remain)
                                    </li>
                                    <li>
                                        The <b>right side</b> (grey background) is the <b>Toxic</b> side (comments that will be deleted)
                                    </li>
                                    <li>
                                        Comment topic area boxes are plotted along the x-axis based on our prediction of your <b>average</b> toxicity rating for comments in that set
                                    </li>
                                </ul>
                            </li>
                
                            <li>
                                The <b>color</b> of the box indicates the <b>system's rating</b> for the same comment topic; you may want to focus on the <b>red-colored boxes</b> that indicate <b>disagreements</b> between "your ratings" and the system's ratings
                            </li>
                        </ul>
                    </Card>
                </div>
            </Cell>
        </LayoutGrid>
        {/if}
    <div class="row">
        <div class="col s8">
            <VegaLite {topic_overview_data} spec={topic_overview_spec} bind:view={topic_overview_view}/>
        </div>
    </div>

    <!-- Old visualizations -->
    <!-- <div style="margin-top: 500px">
        <h6>Overall Performance</h6>
        <div class="row">
            <div class="col s12">
                <div id="overall_perf">
                    <table>
                        <tbody>
                            <tr class="custom-blue">
                                <td class="bold"
                                    >System {data[
                                        "overall_perf"
                                    ]["metric"]} with YOUR labels</td
                                >
                                <td>
                                    <span class="bold-large"
                                        >{data[
                                            "overall_perf"
                                        ]["user_metric"]}</span
                                    >
                                    (Percentile: {data[
                                        "overall_perf"
                                    ]["user_percentile"]})
                                </td>
                            </tr>
                            <tr>
                                <td class="bold"
                                    >System {data[
                                        "overall_perf"
                                    ]["metric"]} with OTHER USERS' labels</td
                                >
                                <td>
                                    <span class="bold-large"
                                        >{data[
                                            "overall_perf"
                                        ]["other_metric"]}</span
                                    >
                                    (95% CI: [{data[
                                        "overall_perf"
                                    ]["other_ci_low"]}, {data[
                                        "overall_perf"
                                    ]["other_ci_high"]}])
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col s8">
                <VegaLite {overall_hist_data} spec={overall_hist_spec} bind:view={overall_hist_view}/>
            </div>
        </div>
        
        <h6>Performance Breakdown</h6>
        <div class="row">
            <div class="col s12">
                <div class="row">
                    <div class="col s12">
                        <VegaLite {class_cond_plot_data} spec={class_cond_plot_spec} bind:view={class_cond_plot_view} />
                    </div>
                </div>
            </div>
        </div>
    </div> -->

</div>
<style>
</style>
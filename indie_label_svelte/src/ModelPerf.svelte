<script lang="ts">
    import { VegaLite } from "svelte-vega";
	import type { View } from "svelte-vega";

    import LayoutGrid, { Cell } from "@smui/layout-grid";
    import Card, { Content } from '@smui/card';

    export let data;

    let perf_plot_spec = data["perf_plot_json"];
    let perf_plot_data = perf_plot_spec["datasets"][
        perf_plot_spec["layer"][0]["data"]["name"]
    ];
    let perf_plot_view: View;

    // let perf_plot2_spec = data["perf_plot2_json"];
    // let perf_plot2_data = perf_plot2_spec["datasets"][perf_plot2_spec["data"]["name"]];
    // let perf_plot2_view: View;
</script>

<div>
    <h6>Your Model Performance</h6>
    <LayoutGrid>
        <Cell span={8}>
            <div class="card-container">
                <Card variant="outlined" padded>
                    <p class="mdc-typography--button"><b>Interpreting your model performance</b></p>
                    <ul>
                        <li>
                            The <b>Mean Absolute Error (MAE)</b> metric indicates the average absolute difference between your model's rating and your actual rating on a held-out set of comments.
                        </li>
                        <li>
                            You want your model to have a <b>lower</b> MAE (indicating <b>less error</b>).
                        </li>
                        <li>
                            <b>Your current MAE: {data["mae"]}</b>
                            <ul>
                                <li>{@html data["mae_status"]}</li>
                                <!-- <li>
                                    This is <b>better</b> (lower) than the average MAE for other users, so your model appears to <b>better capture</b> your views than the typical user model.
                                </li> -->
                            </ul>
                        </li>
                    </ul>
                </Card>
            </div>
        </Cell>
    </LayoutGrid>
    <div>
        <!-- Overall -->
        <!-- <table>
            <tbody>
                <tr>
                    <td>
                        <span class="bold">Mean Absolute Error (MAE)</span><br>
                        
                    </td>
                    <td>
                        <span class="bold-large">{data["mae"]}</span>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="bold">Average rating difference</span><br>
                        This metric indicates the average difference between your model's rating and your actual rating on a held-out set of comments.
                    </td>
                    <td>
                        <span class="bold-large">{data["avg_diff"]}</span>
                    </td>
                </tr>
            </tbody>
        </table> -->

        <!-- Performance visualization -->
        <div>
            <VegaLite {perf_plot_data} spec={perf_plot_spec} bind:view={perf_plot_view}/>
        </div>
    </div>
</div>

<style>
</style>
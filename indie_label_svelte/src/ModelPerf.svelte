<script lang="ts">
    import { VegaLite } from "svelte-vega";
	import type { View } from "svelte-vega";
    import LayoutGrid, { Cell } from "@smui/layout-grid";

    export let data;

    let perf_plot_spec = data["perf_plot_json"];
    let perf_plot_data = perf_plot_spec["datasets"][
        perf_plot_spec["layer"][0]["data"]["name"]
    ];
    let perf_plot_view: View;

</script>

<div>
    <h6>Your Model Performance</h6>
        <ul>
            <li>
                The <b>Mean Absolute Error (MAE)</b> metric indicates the average absolute difference <br>between your model's rating and your actual rating on a held-out set of comments.
            </li>
            <li>
                You want your model to have a <b>lower</b> MAE (indicating <b>less error</b>).
            </li>
            <li>
                <b>Your current MAE: {data["mae"]}</b>
                <ul>
                    <li>{@html data["mae_status"]}</li>
                </ul>
            </li>
        </ul>
    <div>
        <!-- Performance visualization -->
        <div>
            <VegaLite data={perf_plot_data} spec={perf_plot_spec} bind:view={perf_plot_view}/>
        </div>
    </div>
</div>

<style>
</style>
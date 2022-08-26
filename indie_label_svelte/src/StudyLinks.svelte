<script lang="ts">
    import { user } from "./stores/cur_user_store.js";

    let cur_user;
    user.subscribe((value) => {
        cur_user = value;
    });

</script>

<svelte:head>
    <title>Study Links</title>
</svelte:head>

<div class="panel">
    <div class="panel_contents">
        <div>
            <h3>Study Links</h3>
        </div>

        <div>
            <!-- <div class="head_5">{cur_user}</div> -->
            <div class="section_indent">
                <ul>
                    <li>Labeling pages
                        <ul>
                            <li>
                                <a href="http://localhost:5001/?user={cur_user}&tab=labeling&label_mode=3" target="_blank">Group-based model</a>
                            </li>
                            <li>
                                <a href="http://localhost:5001/?user={cur_user}&tab=labeling&label_mode=0" target="_blank">Personalized model</a>
                            </li>
                        </ul>
                    </li>
                    <li>Auditing pages
                        <ul>
                            <li>
                                <a href="http://localhost:5001/?user={cur_user}&scaffold=personal_group" target="_blank">Group-based audit - personal scaffold</a>
                            </li>
                            <li>
                                <a href="http://localhost:5001/?user={cur_user}&scaffold=personal" target="_blank">Individual audit - personal scaffold</a>
                            </li>
                            <li>
                                <a href="http://localhost:5001/?user={cur_user}&scaffold=prompts" target="_blank">Individual audit - prompt scaffold</a>
                            </li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
    </div>
</div>

<style>
    .panel {
        width: 80%;
        padding: 50px;
    }
</style>

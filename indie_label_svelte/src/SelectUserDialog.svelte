<script lang="ts">
    import Dialog, { Title, Content, Actions } from "@smui/dialog";
    import Button, { Label } from "@smui/button";
    import Textfield from "@smui/textfield";
    import Select, { Option } from "@smui/select";
    import { user } from "./stores/cur_user_store.js";
    import { users } from "./stores/all_users_store.js";

    export let open;
    export let cur_user;
    let cur_user_tf = cur_user;
    let cur_user_sel = cur_user;

    let all_users;
    users.subscribe((value) => {
        all_users = value;
    });

    function updateUserTextField() {
        user.update((value) => cur_user_tf);
        if (!all_users.includes(user)) {
            all_users = all_users.concat(cur_user_tf);
            users.update(all_users);
        } 
        open = false;
    }

    function updateUserSel() {
        user.update((value) => cur_user_sel);
        open = false;
    }
</script>

<div>
    <Dialog
        bind:open
        aria-labelledby="simple-title"
        aria-describedby="simple-content"
    >
        <!-- Title cannot contain leading whitespace due to mdc-typography-baseline-top() -->
        <Title id="simple-title">Select Current User</Title>
        <Content id="simple-content">
            <Textfield bind:value={cur_user_tf} label="Enter user's name" />

            <Select bind:value={cur_user_sel} label="Select Menu">
                {#each all_users as u}
                    <Option value={u}>{u}</Option>
                {/each}
            </Select>
        </Content>
        <Actions>
            <Button on:click={updateUserTextField}>
                <Label>Update from TextField</Label>
            </Button>
            <Button on:click={updateUserSel}>
                <Label>Update from Select</Label>
            </Button>
        </Actions>
    </Dialog>
</div>

<style>
    :global(.mdc-dialog__surface) {
        height: 300px;
    }
</style>

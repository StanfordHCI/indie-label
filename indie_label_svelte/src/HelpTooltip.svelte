<script>
    import Tooltip, {
        Wrapper,
        Title,
        Content,
        Link,
        RichActions,
    } from '@smui/tooltip';
    import IconButton from '@smui/icon-button';

    export let text;
</script>

<Tooltip>
    <Content>
        {text}
    </Content>
</Tooltip>
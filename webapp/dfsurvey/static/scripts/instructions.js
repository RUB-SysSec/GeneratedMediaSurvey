$(document).ready(function () {
    $("#continue").click(function () {
        $.ajax({
            type: "PUT",
            url: location.pathname,
        }).done(function (data) {
            console.debug(data["msg"]);
            window.location.href = data["url"];
        });
    });
});
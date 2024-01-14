$(document).ready(function () {
    $("#continue").click(function () {
        $("#introduction_container").addClass("d-none");
        $("#disclaimer_container").removeClass("d-none");
    });

    $("#accept_conditions").click(function () {
        let decision = null;

        $("#consentRadio").find("input[type='radio']:checked").each(function () {
            if (decision !== null) {
                throw Error("Cannot have two decisions!");
            }
            decision = $(this).attr("data-decision");
        });

        if (decision !== null) {
            $("#errorConditions").addClass("d-none");

            data = {
                "decision": decision,
            }

            $.ajax({
                type: "PUT",
                url: location.pathname,
                contentType: 'application/json',
                data: JSON.stringify(data)
            }).done(function (data) {
                console.debug(data["msg"]);
                window.location.href = data["url"];
            });
        } else {
            $("#errorConditions").removeClass("d-none");
        }
    });

});
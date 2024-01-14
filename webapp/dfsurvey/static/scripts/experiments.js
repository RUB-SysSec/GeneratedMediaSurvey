var scaleFn = function () {
    let choice = parseInt($("#scale").val());

    runExperiment(choice);
};

var submitFn = function () {
    try {
        let [_a, _b, _c, choice] = getSelectedChild($("#choices"));
        console.debug("User selected option: " + choice + "!");
        $("#error").addClass("d-none");

        runExperiment(choice);
    }
    catch (err) {
        switch (err.message) {
            case "No answer selected!":
                // mark question read
                $("#error").removeClass("d-none");
                return;
            default:
                // for now throw the error
                throw err;
        }
    }
};

$(document).ready(function () {
    let fn = null;
    if ($("#scale").length) {
        fn = scaleFn;
    } else {
        fn = submitFn;
    }

    $("#submit").click(function () {
        $("#submit").prop("disabled", true);
        fn();
        $("#submit").prop("disabled", false);
    });
});
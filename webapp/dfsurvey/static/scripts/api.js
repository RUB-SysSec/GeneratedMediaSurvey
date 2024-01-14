// force window position at the top everytime
$(window).on('unload', function () {
    $(window).scrollTop(0);
});

history.scrollRestoration = "manual";


//
// Experiments
//
var getSelectedChild = function (element) {
    let selected = new Array();
    $(element).find("input[type='radio']:checked").each(function () {
        let question_id = $(this).attr("data-question-id");
        let option_id = $(this).attr("data-option-id");
        let question_type = $(this).attr("data-question-type");
        let choice = $(this).attr("data-choice");

        selected.push([question_id, option_id, question_type, choice]);
    });

    // make sure we do not overselect or underselect
    if (selected.length == 0) {
        throw Error("No answer selected!");
    } else if (selected.length > 1) {
        throw Error("Selected more than one element!");
    }

    return selected[0];
};

var redirect = function (data) {
    console.debug(data["msg"])
    window.location.href = data["url"];
}

var runExperiment = function (choice) {
    data = {
        "choice": choice,
    }

    // handle possible counter
    var count = parseInt($("#clickCounter").text());
    if (count > 0) {
        data["count"] = count;
    }

    $.ajax({
        type: "POST",
        url: location.pathname,
        contentType: 'application/json',
        data: JSON.stringify(data),
        dataType: "json",
    }).done(function (data) {
        console.debug(data["msg"]);
        if (data["move_on"]) {
            redirect(data);
        } else {
            location.reload();
        }
    });
};
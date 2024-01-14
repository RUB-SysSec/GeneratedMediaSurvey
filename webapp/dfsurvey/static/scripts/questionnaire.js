$(document).ready(function () {
    // setup submit buttons
    $(".submit-category").each(function () {
        let button = $(this);
        $(this).click(function () {
            // find all selected children of this question
            let counter = 0;
            let first = null;
            let results = new Map();
            let category = $(this).attr("data-category");

            $("#questionnaire").find("div.question").each(function () {
                counter += 1;
                let question_type = $(this).attr("data-type");
                let question_id = $(this).attr("data-question-id");

                try {
                    if (question_type == "number" || question_type == "textfield" || question_type == "age") {
                        let tag_name = null;
                        if (question_type == "number" || question_type == "age") {
                            tag_name = "input";
                        } else if (question_type == "textfield") {
                            tag_name = "textarea";
                        }

                        $(this).find(tag_name).each(function () {
                            if (results.has(question_id)) {
                                console.debug(results);
                                throw "Already found an answer for this scale!";
                            }

                            let value = this.value;
                            if (value == "") {
                                throw Error("No answer selected!");
                            }

                            results.set(question_id, {
                                question_type: question_type,
                                value: value,
                            });
                        });
                    } else if (question_type == "scale") {
                        $(this).find("input").each(function () {
                            if (results.has(question_id)) {
                                console.debug(results);
                                throw "Already found an answer for this scale!";
                            }

                            results.set(question_id, {
                                question_type: question_type,
                                value: this.value,
                            });
                        });
                    } else {
                        let [_question_id, option_id, question_type, _] = getSelectedChild(this);

                        console.log(question_id, option_id, question_type);
                        results.set(question_id, {
                            question_type: question_type,
                            option_id: option_id,
                        });
                        console.debug("collected question " + question_id + "!");

                        // remove red text if it got correctly selected
                        if ($(this).hasClass("alert-warning")) {
                            $(this).removeClass("alert-warning");
                        }
                    }
                }
                catch (err) {
                    if (first === null) {
                        first = $(this);
                    }
                    switch (err.message) {
                        case "No answer selected!":
                            // mark question read
                            $(this).addClass("alert-warning");
                            return;
                        default:
                            // for now throw the error
                            throw err;
                    }
                }
            });


            let error = $("#error");
            console.debug("Collected " + results.size + " of " + counter + " answers!");

            if (counter == results.size) {
                // disable error
                error.addClass("d-none");

                // send result
                data = JSON.stringify({
                    category: category,
                    answers: Array.from(results),
                });

                // block button
                button.prop("disabled", true);

                $.ajax({
                    type: "POST",
                    url: location.pathname,
                    contentType: 'application/json',
                    data: data,
                    dataType: "json",
                }).done(function (data) {
                    if (data["redirect"]) {
                        redirect(data);
                    } else {
                        // otherwise just fetch the new version
                        window.location.replace(window.location.href);
                    }
                });
                // unblock button
                button.prop("disabled", false);

            } else {
                error.removeClass("d-none");
                first.get(0).scrollIntoView({ block: "center" });
            }
        });
    });
});
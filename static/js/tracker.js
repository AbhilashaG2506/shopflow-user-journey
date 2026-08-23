// ============================================================
// SHOPFLOW USER JOURNEY TRACKER
// ============================================================


// ============================================================
// DEVICE DETECTION
// ============================================================

function getDeviceType() {

    const userAgent = navigator.userAgent.toLowerCase();

    // Mobile phones
    if (
        /android.*mobile|iphone|ipod|windows phone/.test(userAgent)
    ) {
        return "mobile";
    }

    // Tablets
    if (
        /ipad|android(?!.*mobile)|tablet/.test(userAgent)
    ) {
        return "tablet";
    }

    // Desktop / Laptop
    return "desktop";
}


// ============================================================
// TRACK EVENT
// ============================================================

function trackEvent(eventType, productId = null) {

    const userId =
        localStorage.getItem("shopflow_user_id");

    if (!userId) {

        console.log(
            "No user ID. Event not tracked."
        );

        return;
    }


    fetch("/track-event", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({

            user_id: userId,

            event_type: eventType,

            product_id: productId,

            device: getDeviceType(),

            location: "India",

            traffic_source: "direct"

        })

    })


    .then(response => {

        if (!response.ok) {
            throw new Error(
                "Server returned " + response.status
            );
        }

        return response.json();

    })


    .then(data => {

        console.log(
            "ShopFlow Event:",
            data
        );

    })


    .catch(error => {

        console.error(
            "Tracking error:",
            error
        );

    });

}


// ============================================================
// AUTOMATIC VISIT TRACKING
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        trackEvent("visit");

    }
);
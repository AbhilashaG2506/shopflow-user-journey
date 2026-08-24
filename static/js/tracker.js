
// ============================================================
// SHOPFLOW USER JOURNEY TRACKER
// REAL-TIME E-COMMERCE EVENT TRACKING
// ============================================================


// ============================================================
// DEVICE DETECTION
// ============================================================

function getDeviceType() {

    const userAgent = navigator.userAgent.toLowerCase();

    if (
        /android.*mobile|iphone|ipod|windows phone/.test(userAgent)
    ) {
        return "mobile";
    }

    if (
        /ipad|android(?!.*mobile)|tablet/.test(userAgent)
    ) {
        return "tablet";
    }

    return "desktop";
}


// ============================================================
// TRAFFIC SOURCE
// ============================================================

function getTrafficSource() {

    const referrer = document.referrer.toLowerCase();

    if (!referrer) {
        return "direct";
    }

    if (
        referrer.includes("google.") ||
        referrer.includes("bing.") ||
        referrer.includes("yahoo.")
    ) {
        return "organic_search";
    }

    if (
        referrer.includes("facebook.") ||
        referrer.includes("instagram.") ||
        referrer.includes("twitter.") ||
        referrer.includes("x.com") ||
        referrer.includes("linkedin.") ||
        referrer.includes("youtube.")
    ) {
        return "social";
    }

    return "referral";
}


// ============================================================
// USER ID
// ============================================================

function getShopFlowUserId() {

    let userId = localStorage.getItem("shopflow_user_id");

    if (!userId) {

        userId =
            "LIVE_" +
            Math.random()
                .toString(36)
                .substring(2, 10)
                .toUpperCase();

        localStorage.setItem(
            "shopflow_user_id",
            userId
        );
    }

    return userId;
}


// ============================================================
// TRACK EVENT
// ============================================================

function trackEvent(eventType, productId = null) {

    const eventData = {

        user_id: getShopFlowUserId(),

        event_type: eventType,

        product_id: productId,

        device: getDeviceType(),

        location: "India",

        traffic_source: getTrafficSource(),

        timestamp: new Date().toISOString()
    };


    console.log(
        "SHOPFLOW LIVE EVENT:",
        eventData
    );


    fetch("/track-event", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(eventData)

    })

    .then(response => {

        if (!response.ok) {

            throw new Error(
                "Server returned " +
                response.status
            );

        }

        return response.json();

    })

    .then(data => {

        console.log(
            "SHOPFLOW EVENT SAVED:",
            data
        );

    })

    .catch(error => {

        console.error(
            "SHOPFLOW LIVE TRACKING ERROR:",
            error
        );

    });
}


// ============================================================
// MAKE TRACK EVENT AVAILABLE TO ALL SHOPPING PAGES
// ============================================================

window.trackEvent = trackEvent;


// ============================================================
// AUTOMATIC VISIT TRACKING
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        trackEvent("visit");

    }
);
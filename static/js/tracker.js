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
// TRAFFIC SOURCE DETECTION
// ============================================================

function getTrafficSource() {

    const referrer = document.referrer.toLowerCase();

    // No referrer = direct visit
    if (!referrer) {
        return "direct";
    }

    // Google
    if (referrer.includes("google.")) {
        return "organic_search";
    }

    // Bing
    if (referrer.includes("bing.")) {
        return "organic_search";
    }

    // Yahoo
    if (referrer.includes("yahoo.")) {
        return "organic_search";
    }

    // Social media
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

    // Referral from another website
    return "referral";
}


// ============================================================
// USER ID
// ============================================================

function getShopFlowUserId() {

    let userId = localStorage.getItem(
        "shopflow_user_id"
    );

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

function trackEvent(
    eventType,
    productId = null
) {

    const userId = getShopFlowUserId();

    const eventData = {

        user_id: userId,

        event_type: eventType,

        product_id: productId,

        device: getDeviceType(),

        location: "India",

        traffic_source: getTrafficSource()

    };


    console.log(
        "SHOPFLOW EVENT:",
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
            "EVENT SAVED:",
            data
        );

    })

    .catch(error => {

        console.error(
            "TRACKING ERROR:",
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
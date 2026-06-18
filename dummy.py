import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = [
    "Facebook Ads",
    "Google Ads",
    "TikTok Ads",
    "Unity Ads",
    "Apple Search Ads",
    "Organic",
]
COUNTRIES = ["US", "JP", "KR", "DE", "FR", "BR", "IN"]
DEVICES = ["ios", "android"]


def _clip(values, lower, upper):
    return np.minimum(np.maximum(values, lower), upper)


def _build_campaigns():
    channel_cpi = {
        "Facebook Ads": 3.2,
        "Google Ads": 2.8,
        "TikTok Ads": 2.1,
        "Unity Ads": 1.6,
        "Apple Search Ads": 3.8,
        "Organic": 0.0,
    }
    country_factor = {
        "US": 1.45,
        "JP": 1.35,
        "KR": 1.2,
        "DE": 1.05,
        "FR": 1.0,
        "BR": 0.55,
        "IN": 0.35,
    }
    device_factor = {"ios": 1.15, "android": 0.9}

    rows = []
    for channel in CHANNELS:
        for country in COUNTRIES:
            for device in DEVICES:
                cpi = channel_cpi[channel] * country_factor[country] * device_factor[device]
                rows.append(
                    {
                        "channel": channel,
                        "country": country,
                        "device": device,
                        "cpi": round(cpi, 2),
                    }
                )
    return pd.DataFrame(rows)


def generate_game_ltv_data(num_users=5000, seed=42, max_day=30):
    """Generate synthetic game user data for LTV and UA value analysis."""
    rng = np.random.default_rng(seed)
    campaigns = _build_campaigns()

    user_ids = np.arange(1, num_users + 1)
    channels = rng.choice(
        CHANNELS,
        size=num_users,
        p=[0.24, 0.22, 0.18, 0.16, 0.08, 0.12],
    )
    countries = rng.choice(
        COUNTRIES,
        size=num_users,
        p=[0.24, 0.16, 0.13, 0.12, 0.11, 0.12, 0.12],
    )
    devices = rng.choice(DEVICES, size=num_users, p=[0.46, 0.54])
    install_dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 30, size=num_users), unit="D"
    )

    channel_quality = {
        "Facebook Ads": 1.05,
        "Google Ads": 1.0,
        "TikTok Ads": 0.88,
        "Unity Ads": 0.76,
        "Apple Search Ads": 1.22,
        "Organic": 1.12,
    }
    country_quality = {
        "US": 1.35,
        "JP": 1.25,
        "KR": 1.15,
        "DE": 1.05,
        "FR": 1.0,
        "BR": 0.72,
        "IN": 0.58,
    }
    device_quality = {"ios": 1.12, "android": 0.92}

    quality = np.array(
        [
            channel_quality[channel] * country_quality[country] * device_quality[device]
            for channel, country, device in zip(channels, countries, devices)
        ]
    )
    quality *= rng.lognormal(mean=0.0, sigma=0.35, size=num_users)
    quality = _clip(quality, 0.25, 3.5)

    payer_probability = _clip(0.06 * quality, 0.01, 0.42)
    payer_type = rng.random(num_users) < payer_probability
    whale_probability = np.where(payer_type, _clip(0.015 * quality, 0.002, 0.12), 0)
    is_whale = rng.random(num_users) < whale_probability

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "install_date": install_dates.strftime("%Y-%m-%d"),
            "channel": channels,
            "country": countries,
            "device": devices,
            "user_quality_score": np.round(quality, 4),
            "payer_segment": np.select(
                [is_whale, payer_type],
                ["high_value_payer", "regular_payer"],
                default="non_payer",
            ),
        }
    )

    rows = []
    for idx, user_id in enumerate(user_ids):
        cumulative_revenue = 0.0
        level = int(rng.integers(1, 4))
        base_activity = _clip(quality[idx] * rng.lognormal(0.0, 0.25), 0.15, 4.5)

        for day in range(1, max_day + 1):
            retention_prob = _clip(0.82 * np.exp(-0.065 * (day - 1)) * base_activity, 0.03, 0.98)
            active = int(day == 1 or rng.random() < retention_prob)
            sessions = int(active * rng.poisson(1.4 + 1.1 * base_activity))
            sessions = max(sessions, active)
            playtime = float(active * rng.gamma(2.0 + base_activity, 8.0))
            level += int(active * rng.poisson(0.7 + 0.5 * base_activity))
            ad_impressions = int(active * rng.poisson(3.0 + 1.8 * sessions))
            ad_clicks = int(rng.binomial(ad_impressions, _clip(0.035 * quality[idx], 0.005, 0.18)))

            purchase_chance = 0.006 * quality[idx] * (1.0 + 0.4 * active)
            if payer_type[idx]:
                purchase_chance *= 3.2
            if is_whale[idx]:
                purchase_chance *= 2.5
            purchase_count = int(rng.poisson(purchase_chance))

            if purchase_count > 0:
                amount_scale = 5.0 * quality[idx] * (4.0 if is_whale[idx] else 1.0)
                purchase_amount = float(rng.gamma(1.6, amount_scale, purchase_count).sum())
            else:
                purchase_amount = 0.0

            cumulative_revenue += purchase_amount
            rows.append(
                {
                    "user_id": user_id,
                    "day": day,
                    "active": active,
                    "sessions": sessions,
                    "playtime_minutes": round(playtime, 2),
                    "level_reached": level,
                    "ad_impressions": ad_impressions,
                    "ad_clicks": ad_clicks,
                    "purchase_count": purchase_count,
                    "purchase_amount": round(purchase_amount, 2),
                    "cumulative_revenue": round(cumulative_revenue, 2),
                }
            )

    daily_behavior = pd.DataFrame(rows)
    modeling_dataset = _build_modeling_dataset(users, daily_behavior, campaigns)

    return {
        "users": users,
        "daily_behavior": daily_behavior,
        "campaigns": campaigns,
        "modeling_dataset": modeling_dataset,
    }


def _sum_until(df, column, day):
    return df[df["day"] <= day].groupby("user_id")[column].sum()


def _max_until(df, column, day):
    return df[df["day"] <= day].groupby("user_id")[column].max()


def _build_modeling_dataset(users, daily_behavior, campaigns):
    base = users.merge(campaigns, on=["channel", "country", "device"], how="left")
    by_user = daily_behavior.groupby("user_id")

    features = pd.DataFrame({"user_id": users["user_id"]})
    features["d1_retained"] = _max_until(daily_behavior, "active", 1).reindex(users["user_id"]).fillna(0).astype(int).values
    features["d3_retained"] = _max_until(daily_behavior, "active", 3).reindex(users["user_id"]).fillna(0).astype(int).values
    features["d7_retained"] = _max_until(daily_behavior[daily_behavior["day"] == 7], "active", 7).reindex(users["user_id"]).fillna(0).astype(int).values
    features["sessions_d1"] = _sum_until(daily_behavior, "sessions", 1).reindex(users["user_id"]).fillna(0).astype(int).values
    features["sessions_d3"] = _sum_until(daily_behavior, "sessions", 3).reindex(users["user_id"]).fillna(0).astype(int).values
    features["sessions_d7"] = _sum_until(daily_behavior, "sessions", 7).reindex(users["user_id"]).fillna(0).astype(int).values
    features["playtime_minutes_d7"] = _sum_until(daily_behavior, "playtime_minutes", 7).reindex(users["user_id"]).fillna(0).round(2).values
    features["ad_clicks_d7"] = _sum_until(daily_behavior, "ad_clicks", 7).reindex(users["user_id"]).fillna(0).astype(int).values
    features["purchase_count_d7"] = _sum_until(daily_behavior, "purchase_count", 7).reindex(users["user_id"]).fillna(0).astype(int).values
    features["d7_ltv"] = _sum_until(daily_behavior, "purchase_amount", 7).reindex(users["user_id"]).fillna(0).round(2).values
    features["d30_ltv"] = by_user["purchase_amount"].sum().reindex(users["user_id"]).fillna(0).round(2).values
    features["level_reached_d7"] = _max_until(daily_behavior, "level_reached", 7).reindex(users["user_id"]).fillna(1).astype(int).values
    features["is_payer_d7"] = (features["d7_ltv"] > 0).astype(int)

    modeling = base.merge(features, on="user_id", how="left")
    modeling["d7_roas"] = np.where(modeling["cpi"] > 0, modeling["d7_ltv"] / modeling["cpi"], 0).round(4)
    modeling["d30_roas"] = np.where(modeling["cpi"] > 0, modeling["d30_ltv"] / modeling["cpi"], 0).round(4)
    modeling["profitable_d30"] = (modeling["d30_roas"] >= 1.0).astype(int)

    ordered_columns = [
        "user_id",
        "install_date",
        "channel",
        "country",
        "device",
        "cpi",
        "user_quality_score",
        "payer_segment",
        "d1_retained",
        "d3_retained",
        "d7_retained",
        "sessions_d1",
        "sessions_d3",
        "sessions_d7",
        "playtime_minutes_d7",
        "ad_clicks_d7",
        "purchase_count_d7",
        "is_payer_d7",
        "level_reached_d7",
        "d7_ltv",
        "d30_ltv",
        "d7_roas",
        "d30_roas",
        "profitable_d30",
    ]
    return modeling[ordered_columns]


def save_game_ltv_data(tables, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        df.to_csv(output_path / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic game LTV data.")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory for CSV files.")
    parser.add_argument("--num-users", type=int, default=5000, help="Number of users to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    tables = generate_game_ltv_data(num_users=args.num_users, seed=args.seed)
    save_game_ltv_data(tables, args.output)
    print(f"Generated {len(tables['users'])} users and saved CSV files to {args.output}")


if __name__ == "__main__":
    main()

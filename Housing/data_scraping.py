from homeharvest import scrape_property

zip_codes = list(map(str, range(85001, 85088)))
for zip_code in zip_codes:
    sold_properties = scrape_property(
        location=zip_code,
        listing_type="sold",
        past_days=365*2,
    )
    for_sale_properties = scrape_property(
        location=zip_code,
        listing_type="for_sale",
        past_days=365*2,
    )
    sold_properties.to_csv('homes.csv', mode="a", index=False)
    for_sale_properties.to_csv('homes.csv', mode="a", index=False)
    print(zip_code, len(sold_properties), len(for_sale_properties))

print(properties.head())
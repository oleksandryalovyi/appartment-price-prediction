export const CITY_LABELS: Record<string, string> = {
  bialystok: "Białystok", bydgoszcz: "Bydgoszcz", czestochowa: "Częstochowa", gdansk: "Gdańsk",
  gdynia: "Gdynia", katowice: "Katowice", krakow: "Kraków", lodz: "Łódź", lublin: "Lublin",
  poznan: "Poznań", radom: "Radom", rzeszow: "Rzeszów", szczecin: "Szczecin", warszawa: "Warszawa",
  wroclaw: "Wrocław",
};

export const VALUE_LABELS: Record<string, string> = {
  blockOfFlats: "Block of flats", apartmentBuilding: "Apartment building", tenement: "Tenement",
  condominium: "Full ownership", cooperative: "Cooperative", "udział": "Shared (udział)",
  brick: "Brick", concreteSlab: "Concrete slab", low: "Needs renovation", premium: "Premium",
};

export const DISTANCE_FIELDS: { key: string; label: string; unit: string }[] = [
  { key: "poiCount", label: "POIs within 500 m", unit: "" },
  { key: "schoolDistance", label: "School", unit: "km" },
  { key: "kindergartenDistance", label: "Kindergarten", unit: "km" },
  { key: "clinicDistance", label: "Clinic", unit: "km" },
  { key: "pharmacyDistance", label: "Pharmacy", unit: "km" },
  { key: "postOfficeDistance", label: "Post office", unit: "km" },
  { key: "restaurantDistance", label: "Restaurant", unit: "km" },
  { key: "collegeDistance", label: "College", unit: "km" },
];

export const AMENITIES: { key: string; label: string }[] = [
  { key: "hasBalcony", label: "Balcony" },
  { key: "hasElevator", label: "Elevator" },
  { key: "hasParkingSpace", label: "Parking space" },
  { key: "hasStorageRoom", label: "Storage room" },
  { key: "hasSecurity", label: "Security" },
];

export const label = (v: string) => VALUE_LABELS[v] ?? CITY_LABELS[v] ?? v;

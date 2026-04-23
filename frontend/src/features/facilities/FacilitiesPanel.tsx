import type { EmergencyContact, Facility } from "../../types/analysis";

interface FacilitiesPanelProps {
  facilityType: string;
  specialization: string;
  radiusKm: number;
  locationLat: string;
  locationLng: string;
  facilities: Facility[];
  contacts: EmergencyContact[];
  onFacilityTypeChange: (value: string) => void;
  onSpecializationChange: (value: string) => void;
  onRadiusKmChange: (value: number) => void;
  onLocationLatChange: (value: string) => void;
  onLocationLngChange: (value: string) => void;
  onDetectLocation: () => void;
  onFacilitySearch: (emergencyOnly?: boolean) => void;
}

export function FacilitiesPanel({
  facilityType,
  specialization,
  radiusKm,
  locationLat,
  locationLng,
  facilities,
  contacts,
  onFacilityTypeChange,
  onSpecializationChange,
  onRadiusKmChange,
  onLocationLatChange,
  onLocationLngChange,
  onDetectLocation,
  onFacilitySearch,
}: FacilitiesPanelProps) {
  return (
    <section className="content-card">
      <h2>Facilities & Emergency</h2>
      <div className="row-3">
        <label>
          Facility type
          <select value={facilityType} onChange={(event) => onFacilityTypeChange(event.target.value)}>
            <option value="hospital">Hospital</option>
            <option value="clinic">Clinic</option>
            <option value="pharmacy">Pharmacy</option>
            <option value="emergency">Emergency</option>
          </select>
        </label>
        <label>
          Specialization
          <input value={specialization} onChange={(event) => onSpecializationChange(event.target.value)} placeholder="cardiology" />
        </label>
        <label>
          Radius
          <select value={radiusKm} onChange={(event) => onRadiusKmChange(Number(event.target.value))}>
            <option value={1}>1 km</option>
            <option value={5}>5 km</option>
            <option value={10}>10 km</option>
            <option value={20}>20 km</option>
          </select>
        </label>
      </div>
      <div className="row-3">
        <label>
          Latitude
          <input value={locationLat} onChange={(event) => onLocationLatChange(event.target.value)} />
        </label>
        <label>
          Longitude
          <input value={locationLng} onChange={(event) => onLocationLngChange(event.target.value)} />
        </label>
        <div className="button-stack">
          <button type="button" onClick={onDetectLocation}>
            Use current location
          </button>
          <button type="button" onClick={() => onFacilitySearch(false)}>
            Find facilities
          </button>
          <button type="button" onClick={() => onFacilitySearch(true)}>
            Emergency mode
          </button>
        </div>
      </div>

      <h3>Facilities</h3>
      <div className="facility-list">
        {facilities.length > 0 ? (
          facilities.map((item, idx) => (
            <article key={`${item.provider_name}-${idx}`} className="facility-item">
              <strong>{item.provider_name}</strong>
              {item.is_emergency && (
                <span style={{ marginLeft: "6px", color: "#ef4444", fontSize: "0.8rem", fontWeight: 600 }}>
                  EMERGENCY
                </span>
              )}
              <p>{item.address || "Address unavailable"}</p>
              {item.distance_km != null && (
                <p className="muted" style={{ fontSize: "0.82rem" }}>{item.distance_km.toFixed(1)} km away</p>
              )}
              {item.rating != null && (
                <p className="muted" style={{ fontSize: "0.82rem" }}>Rating: {item.rating} / 5</p>
              )}
              {item.open_now != null && (
                <p style={{ fontSize: "0.82rem", color: item.open_now ? "green" : "#ef4444" }}>
                  {item.open_now ? "Open now" : "Closed"}
                </p>
              )}
              <div style={{ display: "flex", gap: "8px", marginTop: "4px", flexWrap: "wrap" }}>
                {item.maps_url && (
                  <a href={item.maps_url} target="_blank" rel="noreferrer" style={{ fontSize: "0.85rem" }}>
                    Directions
                  </a>
                )}
                {item.phone_number && (
                  <a href={`tel:${item.phone_number}`} style={{ fontSize: "0.85rem" }}>
                    Call {item.phone_number}
                  </a>
                )}
              </div>
            </article>
          ))
        ) : (
          <p className="muted">No facilities loaded yet.</p>
        )}
      </div>

      <h3>Emergency Contacts</h3>
      <div className="facility-list">
        {contacts.map((contact) => (
          <article key={`${contact.name}-${contact.phone_number}`} className="facility-item">
            <strong>{contact.name}</strong>
            <p className="muted">{contact.region}</p>
            <a href={`tel:${contact.phone_number}`} style={{ fontWeight: 600, fontSize: "1rem" }}>
              📞 {contact.phone_number}
            </a>
          </article>
        ))}
      </div>
    </section>
  );
}

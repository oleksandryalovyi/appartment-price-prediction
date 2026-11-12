import { useEffect } from "react";
import L from "leaflet";
import { CircleMarker, MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

const icon = L.icon({
  iconUrl: markerIcon, iconRetinaUrl: markerIcon2x, shadowUrl: markerShadow,
  iconSize: [25, 41], iconAnchor: [12, 41],
});

interface Props {
  centre: [number, number];
  value: [number, number] | null;
  onChange: (p: [number, number]) => void;
}

function Recenter({ centre }: { centre: [number, number] }) {
  const map = useMap();
  useEffect(() => { map.setView(centre, 12); }, [centre[0], centre[1]]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

function ClickHandler({ onChange }: { onChange: Props["onChange"] }) {
  useMapEvents({ click: (e) => onChange([+e.latlng.lat.toFixed(6), +e.latlng.lng.toFixed(6)]) });
  return null;
}

export default function MapPicker({ centre, value, onChange }: Props) {
  return (
    <MapContainer center={centre} zoom={12} className="map" scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <Recenter centre={centre} />
      <ClickHandler onChange={onChange} />
      <CircleMarker center={centre} radius={6} pathOptions={{ color: "#6b7280", fillOpacity: 0.6 }} />
      {value && (
        <Marker position={value} icon={icon} draggable
          eventHandlers={{ dragend: (e) => { const p = e.target.getLatLng(); onChange([+p.lat.toFixed(6), +p.lng.toFixed(6)]); } }} />
      )}
    </MapContainer>
  );
}

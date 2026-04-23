interface ProfilePanelProps {
  busy: boolean;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  age: string;
  gender: string;
  emergencyName: string;
  emergencyPhone: string;
  emergencyEmail: string;
  conditionsText: string;
  allergiesText: string;
  medicationsText: string;
  comorbiditiesText: string;
  pregnancyStatus: string;
  emailVerified?: boolean;
  onFirstNameChange: (value: string) => void;
  onLastNameChange: (value: string) => void;
  onEmailChange: (value: string) => void;
  onPhoneChange: (value: string) => void;
  onAgeChange: (value: string) => void;
  onGenderChange: (value: string) => void;
  onEmergencyNameChange: (value: string) => void;
  onEmergencyPhoneChange: (value: string) => void;
  onEmergencyEmailChange: (value: string) => void;
  onConditionsChange: (value: string) => void;
  onAllergiesChange: (value: string) => void;
  onMedicationsChange: (value: string) => void;
  onComorbiditiesChange: (value: string) => void;
  onPregnancyStatusChange: (value: string) => void;
  onSave: () => void;
  onResendVerification?: () => void;
}

export function ProfilePanel({
  busy,
  firstName,
  lastName,
  email,
  phone,
  age,
  gender,
  emergencyName,
  emergencyPhone,
  emergencyEmail,
  conditionsText,
  allergiesText,
  medicationsText,
  comorbiditiesText,
  pregnancyStatus,
  emailVerified,
  onFirstNameChange,
  onLastNameChange,
  onEmailChange,
  onPhoneChange,
  onAgeChange,
  onGenderChange,
  onEmergencyNameChange,
  onEmergencyPhoneChange,
  onEmergencyEmailChange,
  onConditionsChange,
  onAllergiesChange,
  onMedicationsChange,
  onComorbiditiesChange,
  onPregnancyStatusChange,
  onSave,
  onResendVerification,
}: ProfilePanelProps) {
  return (
    <section className="content-card">
      <h2>Profile</h2>
      <label>
        First name
        <input value={firstName} onChange={(event) => onFirstNameChange(event.target.value)} />
      </label>
      <label>
        Last name
        <input value={lastName} onChange={(event) => onLastNameChange(event.target.value)} />
      </label>
      <label>
        Email
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <input value={email} onChange={(event) => onEmailChange(event.target.value)} style={{ flex: 1 }} />
          {emailVerified === false && (
            <span style={{ color: "#ef4444", fontSize: "0.8rem", whiteSpace: "nowrap" }}>
              Not verified
              {onResendVerification && (
                <button
                  type="button"
                  className="ghost"
                  style={{ marginLeft: "6px", fontSize: "0.78rem", padding: "1px 8px" }}
                  onClick={onResendVerification}
                >
                  Resend
                </button>
              )}
            </span>
          )}
          {emailVerified === true && (
            <span style={{ color: "green", fontSize: "0.8rem" }}>✓ Verified</span>
          )}
        </div>
      </label>
      <label>
        Phone
        <input value={phone} onChange={(event) => onPhoneChange(event.target.value)} />
      </label>
      <div className="row-2">
        <label>
          Age
          <input value={age} onChange={(event) => onAgeChange(event.target.value)} />
        </label>
        <label>
          Gender
          <select value={gender} onChange={(event) => onGenderChange(event.target.value)}>
            <option value="">Unspecified</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
            <option value="na">Prefer not to say</option>
          </select>
        </label>
      </div>
      <label>
        Emergency contact name
        <input value={emergencyName} onChange={(event) => onEmergencyNameChange(event.target.value)} />
      </label>
      <label>
        Emergency contact phone
        <input value={emergencyPhone} onChange={(event) => onEmergencyPhoneChange(event.target.value)} />
      </label>
      <label>
        Emergency contact email (for alert notifications)
        <input
          type="email"
          value={emergencyEmail}
          onChange={(event) => onEmergencyEmailChange(event.target.value)}
          placeholder="emergency@example.com"
        />
      </label>
      <label>
        Conditions (comma separated)
        <input
          value={conditionsText}
          onChange={(event) => onConditionsChange(event.target.value)}
          placeholder="asthma, hypertension"
        />
      </label>
      <label>
        Allergies (comma separated)
        <input
          value={allergiesText}
          onChange={(event) => onAllergiesChange(event.target.value)}
          placeholder="penicillin, peanuts"
        />
      </label>
      <label>
        Medications (comma separated)
        <input
          value={medicationsText}
          onChange={(event) => onMedicationsChange(event.target.value)}
          placeholder="metformin, lisinopril"
        />
      </label>
      <label>
        Comorbidities (comma separated)
        <input
          value={comorbiditiesText}
          onChange={(event) => onComorbiditiesChange(event.target.value)}
          placeholder="diabetes, CKD"
        />
      </label>
      <label>
        Pregnancy status
        <select value={pregnancyStatus} onChange={(event) => onPregnancyStatusChange(event.target.value)}>
          <option value="">Unspecified</option>
          <option value="not_pregnant">Not pregnant</option>
          <option value="possible">Possibly pregnant</option>
          <option value="pregnant">Pregnant</option>
          <option value="unknown">Unknown</option>
        </select>
      </label>
      <button type="button" onClick={onSave} disabled={busy}>
        Save profile
      </button>
    </section>
  );
}
  return (
    <section className="content-card">
      <h2>Profile</h2>
      <label>
        First name
        <input value={firstName} onChange={(event) => onFirstNameChange(event.target.value)} />
      </label>
      <label>
        Last name
        <input value={lastName} onChange={(event) => onLastNameChange(event.target.value)} />
      </label>
      <label>
        Email
        <input value={email} onChange={(event) => onEmailChange(event.target.value)} />
      </label>
      <label>
        Phone
        <input value={phone} onChange={(event) => onPhoneChange(event.target.value)} />
      </label>
      <div className="row-2">
        <label>
          Age
          <input value={age} onChange={(event) => onAgeChange(event.target.value)} />
        </label>
        <label>
          Gender
          <select value={gender} onChange={(event) => onGenderChange(event.target.value)}>
            <option value="">Unspecified</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
            <option value="na">Prefer not to say</option>
          </select>
        </label>
      </div>
      <label>
        Emergency contact name
        <input value={emergencyName} onChange={(event) => onEmergencyNameChange(event.target.value)} />
      </label>
      <label>
        Emergency contact phone
        <input value={emergencyPhone} onChange={(event) => onEmergencyPhoneChange(event.target.value)} />
      </label>
      <label>
        Conditions (comma separated)
        <input
          value={conditionsText}
          onChange={(event) => onConditionsChange(event.target.value)}
          placeholder="asthma, hypertension"
        />
      </label>
      <label>
        Allergies (comma separated)
        <input
          value={allergiesText}
          onChange={(event) => onAllergiesChange(event.target.value)}
          placeholder="penicillin, peanuts"
        />
      </label>
      <label>
        Medications (comma separated)
        <input
          value={medicationsText}
          onChange={(event) => onMedicationsChange(event.target.value)}
          placeholder="metformin, lisinopril"
        />
      </label>
      <label>
        Comorbidities (comma separated)
        <input
          value={comorbiditiesText}
          onChange={(event) => onComorbiditiesChange(event.target.value)}
          placeholder="diabetes, CKD"
        />
      </label>
      <label>
        Pregnancy status
        <select value={pregnancyStatus} onChange={(event) => onPregnancyStatusChange(event.target.value)}>
          <option value="">Unspecified</option>
          <option value="not_pregnant">Not pregnant</option>
          <option value="possible">Possibly pregnant</option>
          <option value="pregnant">Pregnant</option>
          <option value="unknown">Unknown</option>
        </select>
      </label>
      <button type="button" onClick={onSave} disabled={busy}>
        Save profile
      </button>
    </section>
  );
}

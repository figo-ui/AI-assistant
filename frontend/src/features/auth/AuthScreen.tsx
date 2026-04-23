import type { FormEvent } from "react";

type AuthMode = "login" | "register" | "forgot" | "reset";

interface AuthScreenProps {
  mode: AuthMode;
  busy: boolean;
  error: string;
  notice: string;
  // login
  identifier: string;
  password: string;
  // register
  regEmail: string;
  regPassword: string;
  regFirstName: string;
  regLastName: string;
  regPhone: string;
  // forgot password
  forgotEmail: string;
  // reset password
  resetToken: string;
  resetPassword: string;
  resetPasswordConfirm: string;
  onModeChange: (value: AuthMode) => void;
  onIdentifierChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onRegEmailChange: (value: string) => void;
  onRegPasswordChange: (value: string) => void;
  onRegFirstNameChange: (value: string) => void;
  onRegLastNameChange: (value: string) => void;
  onRegPhoneChange: (value: string) => void;
  onForgotEmailChange: (value: string) => void;
  onResetTokenChange: (value: string) => void;
  onResetPasswordChange: (value: string) => void;
  onResetPasswordConfirmChange: (value: string) => void;
  onLogin: (event: FormEvent<HTMLFormElement>) => void;
  onRegister: (event: FormEvent<HTMLFormElement>) => void;
  onForgotPassword: (event: FormEvent<HTMLFormElement>) => void;
  onResetPassword: (event: FormEvent<HTMLFormElement>) => void;
}

export function AuthScreen({
  mode,
  busy,
  error,
  notice,
  identifier,
  password,
  regEmail,
  regPassword,
  regFirstName,
  regLastName,
  regPhone,
  forgotEmail,
  resetToken,
  resetPassword,
  resetPasswordConfirm,
  onModeChange,
  onIdentifierChange,
  onPasswordChange,
  onRegEmailChange,
  onRegPasswordChange,
  onRegFirstNameChange,
  onRegLastNameChange,
  onRegPhoneChange,
  onForgotEmailChange,
  onResetTokenChange,
  onResetPasswordChange,
  onResetPasswordConfirmChange,
  onLogin,
  onRegister,
  onForgotPassword,
  onResetPassword,
}: AuthScreenProps) {
  const isLockout = error.toLowerCase().includes("locked") || error.toLowerCase().includes("too many");

  return (
    <main className="page auth-page">
      <section className="auth-card">
        <h1>AI Healthcare Assistant</h1>
        <p className="muted">Secure conversational guidance with risk scoring and facility support.</p>

        {/* Tab switcher — only show for login/register */}
        {(mode === "login" || mode === "register") && (
          <div className="auth-switch">
            <button type="button" className={mode === "login" ? "active" : ""} onClick={() => onModeChange("login")}>
              Login
            </button>
            <button
              type="button"
              className={mode === "register" ? "active" : ""}
              onClick={() => onModeChange("register")}
            >
              Register
            </button>
          </div>
        )}

        {/* ── Login ── */}
        {mode === "login" && (
          <form className="stack-form" onSubmit={onLogin}>
            <label>
              Email or username
              <input
                value={identifier}
                onChange={(e) => onIdentifierChange(e.target.value)}
                autoComplete="username"
                required
              />
            </label>
            <label>
              Password
              <input
                type="password"
                value={password}
                onChange={(e) => onPasswordChange(e.target.value)}
                autoComplete="current-password"
                required
              />
            </label>
            <button type="submit" disabled={busy}>
              {busy ? "Signing in..." : "Sign in"}
            </button>
            <button
              type="button"
              className="ghost"
              style={{ fontSize: "0.85rem", marginTop: "4px" }}
              onClick={() => onModeChange("forgot")}
            >
              Forgot password?
            </button>
          </form>
        )}

        {/* ── Register ── */}
        {mode === "register" && (
          <form className="stack-form" onSubmit={onRegister}>
            <label>
              First name
              <input value={regFirstName} onChange={(e) => onRegFirstNameChange(e.target.value)} />
            </label>
            <label>
              Last name
              <input value={regLastName} onChange={(e) => onRegLastNameChange(e.target.value)} />
            </label>
            <label>
              Email
              <input
                type="email"
                value={regEmail}
                onChange={(e) => onRegEmailChange(e.target.value)}
                autoComplete="email"
                required
              />
            </label>
            <label>
              Phone
              <input value={regPhone} onChange={(e) => onRegPhoneChange(e.target.value)} />
            </label>
            <label>
              Password
              <input
                type="password"
                value={regPassword}
                onChange={(e) => onRegPasswordChange(e.target.value)}
                autoComplete="new-password"
                required
              />
            </label>
            <p className="muted" style={{ fontSize: "0.8rem" }}>
              Min 8 chars, uppercase, lowercase, and a number.
            </p>
            <button type="submit" disabled={busy}>
              {busy ? "Creating account..." : "Create account"}
            </button>
          </form>
        )}

        {/* ── Forgot Password ── */}
        {mode === "forgot" && (
          <form className="stack-form" onSubmit={onForgotPassword}>
            <p className="muted" style={{ marginBottom: "8px" }}>
              Enter your email and we'll send a reset link.
            </p>
            <label>
              Email
              <input
                type="email"
                value={forgotEmail}
                onChange={(e) => onForgotEmailChange(e.target.value)}
                autoComplete="email"
                required
              />
            </label>
            <button type="submit" disabled={busy}>
              {busy ? "Sending..." : "Send reset link"}
            </button>
            <button
              type="button"
              className="ghost"
              style={{ fontSize: "0.85rem", marginTop: "4px" }}
              onClick={() => onModeChange("login")}
            >
              Back to login
            </button>
          </form>
        )}

        {/* ── Reset Password ── */}
        {mode === "reset" && (
          <form className="stack-form" onSubmit={onResetPassword}>
            <p className="muted" style={{ marginBottom: "8px" }}>
              Enter the token from your email and choose a new password.
            </p>
            <label>
              Reset token
              <input
                value={resetToken}
                onChange={(e) => onResetTokenChange(e.target.value)}
                placeholder="Paste token from email"
                required
              />
            </label>
            <label>
              New password
              <input
                type="password"
                value={resetPassword}
                onChange={(e) => onResetPasswordChange(e.target.value)}
                autoComplete="new-password"
                required
              />
            </label>
            <label>
              Confirm new password
              <input
                type="password"
                value={resetPasswordConfirm}
                onChange={(e) => onResetPasswordConfirmChange(e.target.value)}
                autoComplete="new-password"
                required
              />
            </label>
            <button type="submit" disabled={busy}>
              {busy ? "Resetting..." : "Reset password"}
            </button>
            <button
              type="button"
              className="ghost"
              style={{ fontSize: "0.85rem", marginTop: "4px" }}
              onClick={() => onModeChange("login")}
            >
              Back to login
            </button>
          </form>
        )}

        {/* Account lockout banner */}
        {isLockout && (
          <div
            style={{
              background: "#fef3c7",
              border: "1px solid #f59e0b",
              borderRadius: "6px",
              padding: "10px 14px",
              marginTop: "10px",
              fontSize: "0.88rem",
              color: "#92400e",
            }}
          >
            Your account has been temporarily locked due to too many failed login attempts.
            Please wait 15 minutes or{" "}
            <button
              type="button"
              className="ghost"
              style={{ fontSize: "0.85rem", padding: "0", textDecoration: "underline" }}
              onClick={() => onModeChange("forgot")}
            >
              reset your password
            </button>
            .
          </div>
        )}

        {error && !isLockout && <p className="error">{error}</p>}
        {notice && <p className="status">{notice}</p>}
      </section>
    </main>
  );
}

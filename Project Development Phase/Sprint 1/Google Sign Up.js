import React, { useState, useRef } from "react";
import { useScript } from "./hooks/useScript";
import jwt_decode from "jwt-decode";

const App = () => {
  const googlebuttonref = useRef();
  const [user, setuser] = useState(false);
  const onGoogleSignIn = (user) => {
    let userCred = user.credential;
    let payload = jwt_decode(userCred);
    console.log(payload);
    setuser(payload);
  };
  useScript("https://accounts.google.com/gsi/client", () => {
    window.google.accounts.id.initialize({
      client_id: "1005153530632-ad4qhfn0oieuchej56k0s4cida2ol92e.apps.googleusercontent.com", // here's your Google ID
      callback: onGoogleSignIn,
      auto_select: false,
    });

    window.google.accounts.id.renderButton(googlebuttonref.current, {
      size: "medium",
    });
  });
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
      }}
    >
      {!user && <div ref={googlebuttonref}></div>}
      {user && (
        <div>
          <h1>{user.name}</h1>
          <img src={user.picture} alt="profile" />
          <p>{user.email}</p>

          <button
            onClick={() => {
              setuser(false);
            }}
          >
            Logout
          </button>
        </div>
      )}
    </div>
  );
};

export default App;
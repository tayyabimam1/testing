import { initializeApp } from "firebase/app";
import {getAuth} from "firebase/auth";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCXWs7aFdtOvTD8IBULkBC0V_s_4M6A2B0",
  authDomain: "deepsight-da696.firebaseapp.com",
  projectId: "deepsight-da696",
  storageBucket: "deepsight-da696.firebasestorage.app",
  messagingSenderId: "1066158927363",
  appId: "1:1066158927363:web:386f617086c33c88622836"
};

const firebaseApp = initializeApp(firebaseConfig);
export const firebaseAuth = getAuth(firebaseApp);


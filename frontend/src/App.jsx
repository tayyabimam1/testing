import React from 'react'
import {Routes, Route } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

//Pages
import PredictPage from './pages/PredictPage'
import Hero from './pages/Hero'
import ExtensionPage from './pages/ExtensionPage'
import FAQ from './pages/FAQ'
import LoginPage from './pages/LoginPage'
import SignUpPage from './pages/SignUpPage'

//components
import Header from './components/Header';
import Footer from './components/Footer';

const App = () => {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Hero />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/extension" element={<ExtensionPage />} />
        <Route path="/faq" element={<FAQ />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignUpPage />} />
      </Routes>
      <Footer />
    </>
  )
}

export default App
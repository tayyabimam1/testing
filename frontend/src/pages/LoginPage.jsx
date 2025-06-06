import React, { useState } from 'react'
import { firebaseAuth } from '../firebase/firebase'
import {
  signInWithEmailAndPassword,
  GoogleAuthProvider,
  GithubAuthProvider,
  signInWithPopup
} from 'firebase/auth'
import { useNavigate } from 'react-router-dom'
import Button from 'react-bootstrap/Button'
import Toast from 'react-bootstrap/Toast'
import ToastContainer from 'react-bootstrap/ToastContainer'
import { Link } from 'react-router-dom'

const LoginPage = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showToast, setShowToast] = useState(false)
  const [toastMessage, setToastMessage] = useState('')
  const [toastType, setToastType] = useState('success')

  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()

    try {
      const userCredit = await signInWithEmailAndPassword(firebaseAuth, email, password)
      const user = userCredit.user

      if (!user.emailVerified) {
        setToastMessage('Please verify your email before logging in')
        setToastType('warning')
        setShowToast(true)
        return
      }

      setToastMessage('Login successful!')
      setToastType('success')
      setShowToast(true)
      setTimeout(() => navigate('/'), 3000)
    } catch {
      setToastMessage('Invalid Email or Password')
      setToastType('danger')
      setShowToast(true)
    }
  }

  const signInGoogle = async (e) => {
    e.preventDefault()

    try {
      const provider = new GoogleAuthProvider()
      const results = await signInWithPopup(firebaseAuth, provider)
      const user = results.user
      setToastMessage('Successfully Signed In With Google')
      setToastType('success')
      setShowToast(true)
      setTimeout(() => navigate('/'), 3000)
    } catch {
      setToastMessage('Google Sign In Failed Please Try Later')
      setToastType('danger')
      setShowToast(true)
    }
  }

  const signInGithub = async (e) => {
    e.preventDefault()

    try {
      const gitProvider = new GithubAuthProvider()
      const results = await signInWithPopup(firebaseAuth, gitProvider)
      const user = results.user
      setToastMessage('Successfully Signed In With Github')
      setToastType('success')
      setShowToast(true)
      setTimeout(() => navigate('/'), 3000)
    } catch {
      setToastMessage('Github Sign In Failed Please Try Later')
      setToastType('danger')
      setShowToast(true)
    }
  }

  return (
    <>
      <ToastContainer
        position="top-end"
        style={{
          position: 'fixed',
          top: 20,
          right: 20,
          zIndex: 9999
        }}
      >
        <Toast
          show={showToast}
          onClose={() => setShowToast(false)}
          bg={toastType}
          delay={3000}
          autohide
        >
          <Toast.Header>
            <strong className="me-auto">
              {toastType === 'success'
                ? 'Success!'
                : toastType === 'warning'
                ? 'Warning!'
                : 'Error'}
            </strong>
          </Toast.Header>
          <Toast.Body className={toastType !== 'warning' ? 'text-white' : ''}>
            {toastMessage}
          </Toast.Body>
        </Toast>
      </ToastContainer>

      <div className="custom-bg login-page">
        <div className="container d-flex flex-column align-items-center">
          <h1 className="text-color text-center">Sign In</h1>
          <form
            className="d-flex align-items-center justify-content-center flex-column gap-2 w-50 mx-auto mt-4"
            onSubmit={handleSubmit}
          >
            <label className="text-white">Email</label>
            <input
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{ backgroundColor: '#94A3B8' }}
              placeholder="Enter Your Email"
              className="p-2 w-50 mx-auto"
              type="email"
            />

            <label className="text-white mt-2">Password</label>
            <input
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{ backgroundColor: '#94A3B8' }}
              placeholder="Enter Your Password"
              className="p-2 w-50 mx-auto"
              type="password"
            />

            <Button type="submit" variant="outline-primary" className="mt-4 w-50">
              Login
            </Button>
          </form>

          <div>
            <Button onClick={signInGoogle} variant="danger" className="mt-4 rounded-circle">
              <i className="fa-brands fa-google"></i>
            </Button>
            <Button onClick={signInGithub} variant="dark" className="mt-4 fs-5 ms-2 rounded-circle">
              <i className="fa-brands fa-github"></i>
            </Button>
          </div>

          <hr className="p-2 w-25 mt-4 text-white" />

          <Button className="text-white">
            <Link to="/signup" className="text-decoration-none text-white">
              Sign Up with Email
            </Link>
          </Button>
        </div>
      </div>
    </>
  )
}

export default LoginPage

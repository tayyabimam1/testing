import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { firebaseAuth } from '../firebase/firebase'
import { onAuthStateChanged, signOut } from 'firebase/auth'
import Toast from 'react-bootstrap/Toast'
import ToastContainer from 'react-bootstrap/ToastContainer'

const Header = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [showToast, setShowToast] = useState(false)
  const [toastMessage, setToastMessage] = useState('')
  const [toastType, setToastType] = useState('success')
  const navigate = useNavigate()

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(firebaseAuth, (user) => {
      setIsLoggedIn(!!user)
    })

    return () => unsubscribe()
  }, [])

  const handleLogout = async () => {
    try {
      await signOut(firebaseAuth)
      setToastMessage('Successfully Logged Out')
      setToastType('success')
      setShowToast(true)
      setTimeout(() => navigate('/login'), 3000)
    } catch (error) {
      console.error('Error signing out:', error)
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
                : 'Warning'}
            </strong>
          </Toast.Header>
          <Toast.Body className={toastType !== 'warning' ? 'text-white' : ''}>
            {toastMessage}
          </Toast.Body>
        </Toast>
      </ToastContainer>

    <div className='head-bg d-flex align-items-center justify-content-between px-4 py-3 fixed-top'>
        <div className='d-flex align-items-center'>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" className="text-primary"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"></path></svg>
            <Link to='/' className='text-decoration-none text-white ms-2 fw-bold'>DeepSight</Link>
        </div>

        <div className='d-flex align-items-center'>
            <Link to='/predict' className='text-decoration-none text-white mx-3'>Upload</Link>
            <Link to='/extension' className='text-decoration-none text-white mx-3'>Web Extension</Link>
            <Link to='/faq' className='text-decoration-none text-white mx-3'>FAQ</Link>
            {isLoggedIn ? (
              <button 
                onClick={handleLogout}
                className='text-decoration-none text-white mx-3 btn btn-link'
                style={{ padding: 0, border: 'none', background: 'none' }}
              >
                Logout
              </button>
            ) : (
              <Link to='/login' className='text-decoration-none text-white mx-3'>Login</Link>
            )}
        </div>
    </div>
    </>
  )
}

export default Header